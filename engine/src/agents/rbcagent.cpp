/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: rbcagent.cpp
 * Created on 10.2023
 * @author: meierh
 * 
 */

#include <string>
#include <thread>
#include <fstream>
#include "rbcagent.h"

RBCAgent::RBCAgent
(
    NeuralNetAPI *netSingle,
    vector<unique_ptr<NeuralNetAPI>>& netBatches,
    SearchSettings* searchSettings,
    PlaySettings* playSettings,
    std::string fen,
    RBCAgent::PieceColor selfColor
):
MCTSAgent(netSingle, netBatches, searchSettings, playSettings),
selfColor(selfColor),
currentTurn(1)
{
    //std::cout<<"Create RBCAgent"<<std::endl;
    if(selfColor==PieceColor::empty)
        throw std::invalid_argument("Own color can not be empty");
    
    gen = std::mt19937(rd());
    randomScanDist = std::uniform_int_distribution<unsigned short>(1,6);
    
    cis = std::make_unique<ChessInformationSet>();
    
    FullChessInfo initialGameState(fen);
    if(selfColor==PieceColor::white)
    {
        playerPiecesTracker = initialGameState.white;
        cis->add(initialGameState.black,1.0);
        opponentColor = PieceColor::black;
    }
    else
    {
        playerPiecesTracker = initialGameState.black;
        cis->add(initialGameState.white,1.0);
        opponentColor = PieceColor::white;
    }
    if(cis->validSize()==0)
        throw std::logic_error("Chess information set in empty at creation");
}

RBCAgent::FullChessInfo::FullChessInfo(std::string fen)
{
    using CIS = ChessInformationSet;
    
    std::vector<std::string> fenParts;
    FullChessInfo::splitFEN(fen,fenParts);
    std::string figurePlacementString = fenParts[0];
    std::string activeColorString = fenParts[1];
    std::string castlingString = fenParts[2];
    std::string enPassantString = fenParts[3];
    std::string halfMoveClockString = fenParts[4];
    std::string fullMoveNumberString = fenParts[5];

    std::array<std::pair<ChessInformationSet::PieceType,RBCAgent::PieceColor>,64> figureBoard = FullChessInfo::decodeFENFigurePlacement(figurePlacementString);
    for(uint index=0; index<figureBoard.size(); index++)
    {
        std::pair<CIS::PieceType,RBCAgent::PieceColor>& sqInfo = figureBoard[index];
        
        CIS::OnePlayerChessInfo* colorSide;        
        if(sqInfo.second==RBCAgent::PieceColor::white)
            colorSide = &(this->white);
        else if(sqInfo.second==RBCAgent::PieceColor::black)
            colorSide = &(this->black);
        else
            continue;
        
        CIS::Square sq = CIS::boardIndexToSquare(index);
        switch (sqInfo.first)
        {
            case CIS::PieceType::pawn:
                colorSide->pawns.push_back(sq);
                break;
            case CIS::PieceType::knight:
                colorSide->knights.push_back(sq);
                break;
            case CIS::PieceType::bishop:
                colorSide->bishops.push_back(sq);
                break;
            case CIS::PieceType::rook:
                colorSide->rooks.push_back(sq);
                break;
            case CIS::PieceType::queen:
                colorSide->queens.push_back(sq);
                break;
            case CIS::PieceType::king:
                colorSide->kings.push_back(sq);
                break;
            default:
                break;
        }
    }
    
    if(activeColorString=="w")
        this->nextTurn = PieceColor::white;
    else if(activeColorString=="b")
        this->nextTurn = PieceColor::black;
    else
        throw std::logic_error("Invalid character of active color string");
    
    this->white.kingside=false;
    this->white.queenside=false;
    this->black.kingside=false;
    this->black.queenside=false;
    //std::cout<<"castlingString:"<<castlingString<<std::endl;
    for(char c : castlingString)
    {
        //std::cout<<" "<<c<<std::endl;
        switch (c)
        {
            case 'K':
                this->white.kingside=true;
                break;
            case 'Q':
                this->white.queenside=true;
                break;
            case 'k':
                this->black.kingside=true;
                break;
            case 'q':
                this->black.queenside=true;
                break;
            case '-':
                break;
            default:
                throw std::logic_error("Invalid character in Castling string");
        }
    }
    
    if(enPassantString!="-")
    {
        char colChar = enPassantString[0];
        int colInt = colChar-'a';
        char rowChar = enPassantString[1];
        int rowInt = rowChar-'1';
        if(this->nextTurn == PieceColor::white)
        {
            this->black.en_passant = CIS::Square(colInt,rowInt);
            this->black.en_passant_valid = true;
            this->white.en_passant_valid = false;
        }
        else
        {
            this->white.en_passant = CIS::Square(colInt,rowInt);
            this->white.en_passant_valid = true;
            this->black.en_passant_valid = false;
        }
    }
    else if(enPassantString=="-")
    {
        this->white.en_passant_valid = false;
        this->black.en_passant_valid = false;
    }
    else
        throw std::logic_error("Invalid character of castling string");
    
    std::uint8_t no_progress_count = std::stoi(halfMoveClockString);
    this->white.no_progress_count = no_progress_count;
    this->black.no_progress_count = no_progress_count;
    
    this->nextCompleteTurn = std::stoi(fullMoveNumberString);
    
    /*
    std::cout<<"this->black.kingside:"<<this->black.kingside<<std::endl;
    std::cout<<"this->black.queenside:"<<this->black.queenside<<std::endl;
    std::cout<<"this->white.kingside:"<<this->white.kingside<<std::endl;
    std::cout<<"this->white.queenside:"<<this->white.queenside<<std::endl;
    
    std::cout<<"this->white:"<<this->white.to_string()<<std::endl;
    std::cout<<"this->black:"<<this->black.to_string()<<std::endl;
    */
}

open_spiel::chess::Color RBCAgent::AgentColor_to_OpenSpielColor
(
    const RBCAgent::PieceColor agent_pC
)
{
    open_spiel::chess::Color result;
    switch(agent_pC)
    {
        case PieceColor::black:
            result = open_spiel::chess::Color::kBlack;
            break;
        case PieceColor::white:
            result = open_spiel::chess::Color::kWhite;
            break;
        case PieceColor::empty:
            result = open_spiel::chess::Color::kEmpty;
            break;
        default:
            throw std::logic_error("Conversion failure from RBCAgent::PieceColor to open_spiel::chess::Color!");
    }
    return result;
}

RBCAgent::PieceColor RBCAgent::OpenSpielColor_to_RBCColor
(
    const open_spiel::chess::Color os_pC
)
{
    PieceColor result;
    switch(os_pC)
    {
        case open_spiel::chess::Color::kBlack:
            result = PieceColor::black;
            break;        
        case open_spiel::chess::Color::kWhite:
            result = PieceColor::white;
            break;        
        case open_spiel::chess::Color::kEmpty:
            result = PieceColor::empty;
            break;
        default:
            throw std::logic_error("Conversion failure from open_spiel::chess::Color to RBCAgent::PieceColor!");
    }
    return result;
}

open_spiel::rbc::MovePhase RBCAgent::AgentPhase_to_OpenSpielPhase
(
    const RBCAgent::MovePhase agent_mP
)
{
    open_spiel::rbc::MovePhase result;
    switch(agent_mP)
    {
        case MovePhase::Sense:
            result = open_spiel::rbc::MovePhase::kSensing;
            break;        
        case MovePhase::Move:
            result = open_spiel::rbc::MovePhase::kMoving;
            break;
        default:
            throw std::logic_error("Conversion failure from RBCAgent::MovePhase to open_spiel::rbc::MovePhase!");
    }
    return result;
}

RBCAgent::MovePhase RBCAgent::OpenSpielPhase_to_RBCPhase
(
    const open_spiel::rbc::MovePhase os_mP
)
{
    RBCAgent::MovePhase result;
    switch(os_mP)
    {
        case open_spiel::rbc::MovePhase::kSensing:
            result = MovePhase::Sense;
            break;        
        case open_spiel::rbc::MovePhase::kMoving:
            result = MovePhase::Move;
            break;
        default:
            throw std::logic_error("Conversion failure from RBCAgent::MovePhase to open_spiel::rbc::MovePhase!");
    }
    return result;
}

std::pair<std::array<std::uint8_t,9>,std::array<RBCAgent::CIS::Square,9>> RBCAgent::getSensingBoardIndexes(RBCAgent::CIS::Square sq)
{
    std::bitset<8> senseSquaresValid;
    
    std::pair<std::array<std::uint8_t,9>,std::array<CIS::Square,9>> result;
    std::array<std::uint8_t,9>& senseBoardIndexes = result.first;
    std::array<CIS::Square,9>& senseBoardSquares = result.second;
    
    senseBoardIndexes[0] = CIS::squareToBoardIndex(sq);
    senseBoardSquares[0] = sq;
    
    senseSquaresValid[0] = sq.vertPlus(1);
    senseBoardIndexes[1] = CIS::squareToBoardIndex(sq);
    senseBoardSquares[1] = sq;
    
    senseSquaresValid[1] = sq.horizPlus(1);
    senseBoardIndexes[2] = CIS::squareToBoardIndex(sq);
    senseBoardSquares[2] = sq;
    
    for(uint i=0; i<2; i++)
    {
        senseSquaresValid[i+2] = sq.vertMinus(1);
        senseBoardIndexes[i+3] = CIS::squareToBoardIndex(sq);
        senseBoardSquares[i+3] = sq;
    }
    
    for(uint i=0; i<2; i++)
    {
        senseSquaresValid[i+4] = sq.horizMinus(1);
        senseBoardIndexes[i+5] = CIS::squareToBoardIndex(sq);
        senseBoardSquares[i+5] = sq;
    }
    
    for(uint i=0; i<2; i++)
    {
        senseSquaresValid[i+6] = sq.vertPlus(1);
        senseBoardIndexes[i+7] = CIS::squareToBoardIndex(sq);
        senseBoardSquares[i+7] = sq;
    }   
    
    if(!senseSquaresValid.all())
        throw std::logic_error("Invalid sensing area");
    return result;    
}

std::string RBCAgent::FullChessInfo::getFEN
(
    const CIS::OnePlayerChessInfo& white,
    const CIS::OnePlayerChessInfo& black,
    const PieceColor nextTurn,
    const unsigned int nextCompleteTurn
)
{
    std::array<const CIS::OnePlayerChessInfo*,2> colors = {&white,&black};
    
    std::array<std::array<char,8>,8> chessBoard;
    std::for_each(chessBoard.begin(),chessBoard.end(),[](auto& row){row.fill(' ');});
    std::string castlingString;
    std::string enPassantString;
    unsigned int halfTurns=-1;
    
    unsigned int charOffset=0;
    for(const CIS::OnePlayerChessInfo* oneColorInfoPtr : colors)
    {
        const CIS::OnePlayerChessInfo& oneColorInfo = *oneColorInfoPtr;
        
        std::vector<std::tuple<const std::vector<CIS::Square>*,char>> piecesList =
            {
            {&(oneColorInfo.pawns),  'P'},
            {&(oneColorInfo.knights),'N'},
            {&(oneColorInfo.bishops),'B'},
            {&(oneColorInfo.rooks),  'R'},
            {&(oneColorInfo.queens), 'Q'},
            {&(oneColorInfo.kings),  'K'}
            };
            
        for(const std::tuple<const std::vector<CIS::Square>*,char>& onePieceType : piecesList)
        {
            const std::vector<CIS::Square>* squareList = std::get<0>(onePieceType);
            char symbolToPrint = std::get<1>(onePieceType);
            for(const CIS::Square& sq : *squareList)
            {
                unsigned int column = static_cast<unsigned int>(sq.column);
                unsigned int row = static_cast<unsigned int>(sq.row);
                
                char& squareSymbol = chessBoard[row][column];
                if(squareSymbol != ' ')
                    throw std::logic_error("Pieces overlay!");
                squareSymbol = symbolToPrint + charOffset;
            }
        }
        
        if(oneColorInfo.kingside)
            castlingString += 'K'+charOffset;
        if(oneColorInfo.queenside)
            castlingString += 'Q'+charOffset;
        
        charOffset+=32;
    }
    
    if(nextTurn==PieceColor::white)
    {
        if(black.en_passant_valid)
        {
            enPassantString += black.en_passant.to_string();
        }
    }
    else if(nextTurn==PieceColor::black)
    {
        if(white.en_passant_valid)
        {
            enPassantString += white.en_passant.to_string();
        }
    }
    else
        throw std::logic_error("Empty color can not happen!"); 
    
    halfTurns = std::min(white.no_progress_count,black.no_progress_count);
        
    //Build string
    std::string piecePlacement;
    std::vector<std::string> piecePlacementRows;
    for(std::int8_t row=7; row>=0; --row)
    {
        const std::array<char,8>& oneRow = chessBoard[row];
        std::string oneRowStr;
        std::queue<char> oneRowQ;
        std::for_each(oneRow.begin(),oneRow.end(),[&](char entry){oneRowQ.push(entry);});
        unsigned int counter=0;
        while(!oneRowQ.empty())
        {
            char curr = oneRowQ.front();
            oneRowQ.pop();
            if(curr==' ')
            {
                counter++;
            }
            else
            {
                if(counter>0)
                    oneRowStr += std::to_string(counter);
                oneRowStr += curr;
                counter = 0;
            }
        }
        if(counter>0)
            oneRowStr += std::to_string(counter);
        piecePlacementRows.push_back(oneRowStr);
    }
    for(unsigned int row=0; row<piecePlacementRows.size()-1; row++)
    {
        piecePlacement += piecePlacementRows[row]+'/';
    }
    piecePlacement += piecePlacementRows.back();
        
    std::string activeColor = (nextTurn==PieceColor::white)?"w":"b";
    
    std::string castlingAvail = (castlingString.size()>0)?castlingString:"-";
    
    std::string enPassantAvail = (enPassantString.size()>0)?enPassantString:"-";
    
    std::string halfmoveClock = std::to_string(halfTurns);
    std::string fullmoveClock = std::to_string(nextCompleteTurn);
    
    std::string fen = piecePlacement+' '+activeColor+' '+castlingAvail+' '
                     +enPassantAvail+' '+halfmoveClock+' '+fullmoveClock;
    return fen;
}

void RBCAgent::FullChessInfo::splitFEN
(
    std::string fen,
    std::vector<std::string>& fenParts
)
{
    using CIS = ChessInformationSet;
    fenParts.resize(6);
    
    // Full Move
    //std::cout<<"|"<<fen<<"|"<<std::endl;
    if(fen.back()==' ')
        throw std::invalid_argument("Invalid fen string ending 6");
    uint index = fen.rfind(' ');
    fenParts[5] = fen.substr(index+1,fen.size()-index);
    fen = fen.substr(0,index);
    try
    {std::stoi(fenParts[5]);}
    catch(...)
    {throw std::invalid_argument("Invalid total move string");}
    
    //Half move
    //std::cout<<"|"<<fen<<"|"<<std::endl;
    if(fen.back()==' ')
        throw std::invalid_argument("Invalid fen string ending 5");
    index = fen.rfind(' ');
    fenParts[4] = fen.substr(index+1,fen.size()-index);
    fen = fen.substr(0,index);
    try
    {std::stoi(fenParts[4]);}
    catch(...)
    {throw std::invalid_argument("Invalid half move string");}
    
    //En_passant
    //std::cout<<"|"<<fen<<"|"<<std::endl;
    if(fen.back()==' ')
        throw std::invalid_argument("Invalid fen string ending 4");
    index = fen.rfind(' ');
    fenParts[3] = fen.substr(index+1,fen.size()-index);
    fen = fen.substr(0,index);
    if(fenParts[3].size()>2 || fenParts[3].size()<1)
        throw std::invalid_argument("Invalid en_passant string size");
    if(fenParts[3].find(' ')!=std::string::npos)
        throw std::invalid_argument("Invalid en_passant string character");
    
    //Castling
    //std::cout<<"|"<<fen<<"|"<<std::endl;
    if(fen.back()==' ')
        throw std::invalid_argument("Invalid fen string ending 3");
    index = fen.rfind(' ');
    fenParts[2] = fen.substr(index+1,fen.size()-index);
    fen = fen.substr(0,index);
    if(fenParts[2].size()>4 || fenParts[2].size()<1)
        throw std::invalid_argument("Invalid castling string");
    if(fenParts[2].find(' ')!=std::string::npos)
        throw std::invalid_argument("Invalid castling string character");
    
    //Active player
    //std::cout<<"|"<<fen<<"|"<<std::endl;
    if(fen.back()==' ')
        throw std::invalid_argument("Invalid fen string ending 2");
    index = fen.rfind(' ');
    fenParts[1] = fen.substr(index+1,fen.size()-index);
    fen = fen.substr(0,index);
    if(fenParts[1].size()!=1)
        throw std::invalid_argument("Invalid active player string");
    if(fenParts[1].find(' ')!=std::string::npos)
        throw std::invalid_argument("Invalid active player string character");
    
    //Piece placement
    //std::cout<<"|"<<fen<<"|"<<std::endl;
    if(fen.back()==' ')
        throw std::invalid_argument("Invalid fen string ending 1");
    fenParts[0] = fen;
}

void RBCAgent::splitObsFEN
(
    std::string obsFen,
    std::vector<std::string>& obsFenParts
) const
{
    using CIS = ChessInformationSet;
    obsFenParts.resize(6);
    
    // Full Move
    //std::cout<<"|"<<fen<<"|"<<std::endl;
    if(obsFen.back()==' ')
        throw std::invalid_argument("Invalid obsFen string ending 6");
    uint index = obsFen.rfind(' ');
    obsFenParts[5] = obsFen.substr(index+1,obsFen.size()-index);
    obsFen = obsFen.substr(0,index);
    if(obsFenParts[5].size()!=1)
        throw std::invalid_argument("Invalid illegal move string");
    if(obsFenParts[5].find(' ')!=std::string::npos)
        throw std::invalid_argument("Invalid illegal move string character");
    
    //Half move
    //std::cout<<"|"<<obsFen<<"|"<<std::endl;
    if(obsFen.back()==' ')
        throw std::invalid_argument("Invalid obsFen string ending 5");
    index = obsFen.rfind(' ');
    obsFenParts[4] = obsFen.substr(index+1,obsFen.size()-index);
    obsFen = obsFen.substr(0,index);
    if(obsFenParts[4].size()!=1)
        throw std::invalid_argument("Invalid side to play string");
    if(obsFenParts[4].find(' ')!=std::string::npos)
        throw std::invalid_argument("Invalid side to play string character");
    
    //En_passant
    //std::cout<<"|"<<obsFen<<"|"<<std::endl;
    if(obsFen.back()==' ')
        throw std::invalid_argument("Invalid obsFen string ending 4");
    index = obsFen.rfind(' ');
    obsFenParts[3] = obsFen.substr(index+1,obsFen.size()-index);
    obsFen = obsFen.substr(0,index);
    if(obsFenParts[3].size()!=1)
        throw std::invalid_argument("Invalid capture string");
    if(obsFenParts[3].find(' ')!=std::string::npos)
        throw std::invalid_argument("Invalid capture string character");
    
    //Castling
    //std::cout<<"|"<<obsFen<<"|"<<std::endl;
    if(obsFen.back()==' ')
        throw std::invalid_argument("Invalid obsFen string ending 3");
    index = obsFen.rfind(' ');
    obsFenParts[2] = obsFen.substr(index+1,obsFen.size()-index);
    obsFen = obsFen.substr(0,index);
    if(obsFenParts[2].size()!=1)
        throw std::invalid_argument("Invalid phase string");
    if(obsFenParts[2].find(' ')!=std::string::npos)
        throw std::invalid_argument("Invalid phase string character");
    
    //Active player
    //std::cout<<"|"<<obsFen<<"|"<<std::endl;
    if(obsFen.back()==' ')
        throw std::invalid_argument("Invalid obsFen string ending 2");
    index = obsFen.rfind(' ');
    obsFenParts[1] = obsFen.substr(index+1,obsFen.size()-index);
    obsFen = obsFen.substr(0,index);
    if(obsFenParts[2].size()>2 || obsFenParts[2].size()<1)
        throw std::invalid_argument("Invalid castling string");
    if(obsFenParts[2].find(' ')!=std::string::npos)
        throw std::invalid_argument("Invalid castling string character");
    
    //Piece placement
    //std::cout<<"|"<<obsFen<<"|"<<std::endl;
    if(obsFen.back()==' ')
        throw std::invalid_argument("Invalid obsFen string ending 1");
    obsFenParts[0] = obsFen;
}

void RBCAgent::set_search_settings
(
    StateObj *rbcState,
    SearchLimits *searchLimits,
    EvalInfo* evalInfo
)
{
    this->rbcState = rbcState;
    
    //Reduce hypotheses using the previous move information
    if(!(currentTurn==1 && selfColor==PieceColor::white))
    {
        std::cout<<"------Reduce hypotheses using the previous move information--------"<<selfColor<<std::endl;
        handleOpponentMoveInfo(rbcState);
        std::cout<<"-----------------------Forward hypotheses--------------------------"<<selfColor<<std::endl;
        stepForwardHypotheses();
    }
    
    //Scan the board an reduce hypotheses
    std::cout<<"------------------Scan the board and reduce hypotheses------------------"<<selfColor<<std::endl;
    ChessInformationSet::Square scanCenter = applyScanAction(rbcState);
    handleScanInfo(rbcState,scanCenter);
            
    // setup MCTS search
    std::cout<<"----------------------------MCTS setup----------------------------------"<<selfColor<<std::endl;
    this->evalInfo = evalInfo;
    StateObj* chessState = setupMoveSearchState();
    MCTSAgent::set_search_settings(chessState,searchLimits,evalInfo);
}

void RBCAgent::perform_action()
{
    // Run mcts tree and set action to game
    std::cout<<"----------------------------Run action----------------------------------"<<selfColor<<std::endl;
    //throw std::runtime_error("Temp Stop");
    MCTSAgent::perform_action();
    delete this->state;
        
    //Reduce hypotheses using the own move information
    std::cout<<"--------------------Handle action move information----------------------"<<selfColor<<std::endl;
    handleSelfMoveInfo(rbcState);
}

void RBCAgent::completeMoveData
(
    open_spiel::chess::Move& move,
    ChessInformationSet::OnePlayerChessInfo& opponentInfo
) const
{
    using CIS = ChessInformationSet;
    std::function<std::pair<bool,CIS::PieceType>(const ChessInformationSet::Square&)> opponentSquareToPiece = opponentInfo.getSquarePieceTypeCheck();
    
    CIS::Square from(move.from);
    CIS::Square to(move.to);
    auto [nonEmpty,piece] = opponentSquareToPiece(from);
    if(!nonEmpty)
    {
        throw std::logic_error("Move from empty square not possible");
    }
    if(piece==CIS::PieceType::empty || piece==CIS::PieceType::unknown)
    {
        throw std::logic_error("Move from square with empty or unknown type");
    }
    open_spiel::chess::Piece movedPiece
    {
        AgentColor_to_OpenSpielColor(opponentColor),
        CIS::CISPieceType_to_OpenSpielPieceType(piece)
    };
    move.piece = movedPiece;
    
    if(piece==CIS::PieceType::king)
    {
        std::pair<std::int8_t,std::int8_t> diffFromTo = from.diffToSquare(to);
        if(diffFromTo.first==1 || diffFromTo.second==1)
        {
            move.is_castling = false;
        }
        else if(diffFromTo.first==2)
        {
            move.is_castling = true;
        }
        else
            throw std::logic_error("King can not move three or more squares");
    }
}

std::unique_ptr<std::vector<std::pair<ChessInformationSet::OnePlayerChessInfo,double>>> RBCAgent::generateHypotheses
(
    ChessInformationSet::OnePlayerChessInfo& piecesOpponent,
    ChessInformationSet::OnePlayerChessInfo& piecesSelf,
    const RBCAgent::PieceColor selfColor
) const
{
    //std::cout<<"Player:"<<selfColor<<" generates Hypotheses of state of player:"<<opponentColor<<std::endl;
    
    if(selfColor == PieceColor::empty)
        throw std::invalid_argument("selfColor must not be PieceColor::empty");
    
    using CIS_Square = ChessInformationSet::Square;
    using CIS_CPI = ChessInformationSet::OnePlayerChessInfo;
    auto hypotheses = std::make_unique<std::vector<std::pair<CIS_CPI,double>>>();
    
    FullChessInfo fullState;
    unsigned int nextCompleteTurn;
    
    // switch positions to get all legal actions
    if(selfColor == PieceColor::white)
    {
        fullState.white = piecesSelf;
        fullState.black = piecesOpponent;
        fullState.nextTurn = PieceColor::black;
        fullState.nextCompleteTurn = currentTurn;
    }
    else
    {
        fullState.white = piecesOpponent;
        fullState.black = piecesSelf;
        fullState.nextTurn = PieceColor::white;
        fullState.nextCompleteTurn = currentTurn+1;
    }
    std::string fen = fullState.getFEN();
    //std::cout<<"Construct state for possible moves:"<<fen<<std::endl;
    OpenSpielState hypotheticState(open_spiel::gametype::SupportedOpenSpielVariants::RBC);
    hypotheticState.set(fen,false,open_spiel::gametype::SupportedOpenSpielVariants::RBC);
    hypotheticState.do_action(0); // Dummy sense to get to moving phase
    if(OpenSpielColor_to_RBCColor(hypotheticState.currentPlayer())!=opponentColor)
    {
        throw std::logic_error("Hypothetic state has invalid player color!");
    }
    if(OpenSpielPhase_to_RBCPhase(hypotheticState.currentPhase())!=MovePhase::Move)
    {
        throw std::logic_error("Hypothetic state has invalid move phase!");
    }
    open_spiel::chess::Color os_opponent = AgentColor_to_OpenSpielColor(opponentColor);
    std::vector<Action> legal_actions_int = hypotheticState.legal_actions();
    std::vector<std::pair<open_spiel::chess::Move,bool>> legal_actions_move(legal_actions_int.size());
    for(unsigned int actionInd=0; actionInd<legal_actions_int.size(); actionInd++)
    {
        std::tuple<std::uint8_t,open_spiel::chess::Move,bool> move = hypotheticState.ActionToIncompleteMove(legal_actions_int[actionInd],os_opponent);
        MovePhase actionPhaseType = static_cast<MovePhase>(std::get<0>(move));
        open_spiel::chess::Move& moveData = std::get<1>(move);
        bool passMove =std::get<2>(move);
        if(actionPhaseType == MovePhase::Sense)
        {
            throw std::logic_error("Invalid phase action");
        }
        if(!passMove)
        {
            completeMoveData(moveData,piecesOpponent);
        }
        legal_actions_move[actionInd] = {moveData,passMove};
    }
    
    for(const std::pair<open_spiel::chess::Move,bool>& legal_action : legal_actions_move)
    {
        const open_spiel::chess::Move& move = legal_action.first;
        bool passMove = legal_action.second;
        
        if(passMove)
        {
            hypotheses->push_back({piecesOpponent,0});
        }
        else
        {
            CIS::Square from(move.from);
            CIS::Square to(move.to);
            CIS::PieceType pieceType = CIS::OpenSpielPieceType_to_CISPieceType(move.piece.type);
            PieceColor moveColor = OpenSpielColor_to_RBCColor(move.piece.color);
            CIS::PieceType promPieceType = CIS::OpenSpielPieceType_to_CISPieceType(move.promotion_type);
            bool castling = move.is_castling;
            
            hypotheses->push_back({piecesOpponent,0});
            CIS::OnePlayerChessInfo& new_hypothese = hypotheses->back().first;
            
            // test for color match
            if(moveColor!=opponentColor)
                throw std::logic_error("Opponent move color mismatch!");
            
            // Correctness checks
            if(promPieceType==CIS::PieceType::empty)
            {
                bool isPromotion=false;
                if(pieceType==CIS::PieceType::pawn)
                {
                    if(moveColor==PieceColor::white)
                    {
                        if(to.row==CIS::ChessRow::eight)
                            isPromotion=true;
                    }
                    else
                    {
                        if(to.row==CIS::ChessRow::one)
                            isPromotion=true;
                    }
                }
                if(isPromotion)
                    throw std::logic_error("Error in received legal move: Pawn at end but no promotion!");
            }
            if(promPieceType!=CIS::PieceType::empty)
            {
                if(castling)
                    throw std::logic_error("Castling and Promotion in one move not possible!");
                if(pieceType!=CIS::PieceType::pawn)
                    throw std::logic_error("Non pawn piece can not be promoted!");
                if(promPieceType==CIS::PieceType::pawn || promPieceType==CIS::PieceType::king)
                    throw std::logic_error("Piece can not be promoted to a pawn or a king!");
                if(pieceType==CIS::PieceType::pawn)
                {
                    if(moveColor==PieceColor::white)
                    {
                        if(to.row!=CIS::ChessRow::eight || from.row!=CIS::ChessRow::seven)
                            throw std::logic_error("False from and to squares for promotion!");
                    }
                    else
                    {
                        if(to.row!=CIS::ChessRow::one || from.row!=CIS::ChessRow::two)
                            throw std::logic_error("False from and to squares for promotion!");
                    }
                }
            }
            
            new_hypothese.applyMove(from,to,pieceType,promPieceType,castling);
        }
        //std::cout<<"Generated hypothese: "<<hypotheses->back().first.to_string()<<std::endl;
    }
    return hypotheses;
}

std::unique_ptr<std::vector<std::pair<ChessInformationSet::OnePlayerChessInfo,double>>> RBCAgent::generateHypotheses
(
    ChessInformationSet::OnePlayerChessInfo& piecesOpponent
)
{
    return generateHypotheses(piecesOpponent,this->playerPiecesTracker,this->selfColor);
}

std::array<std::pair<ChessInformationSet::PieceType,RBCAgent::PieceColor>,64> RBCAgent::FullChessInfo::decodeFENFigurePlacement
(
    std::string figurePlacementFEN
)
{
    using CIS = ChessInformationSet;
    std::unordered_set<char> allowedCharsInRowPieces = {'1','2','3','4','5','6','7','8',' ',
                                                        'p','r','n','b','q','k',
                                                        'P','R','N','B','Q','K'};

    std::vector<std::string> figurePlacementFENParts;
    std::string::size_type index;
    while((index=figurePlacementFEN.find("/"))!=std::string::npos)
    {
        figurePlacementFENParts.push_back(figurePlacementFEN.substr(0,index));
        figurePlacementFEN = figurePlacementFEN.substr(index+1);
    }
    figurePlacementFENParts.push_back(figurePlacementFEN);
    if(figurePlacementFENParts.size()!=8)
        throw std::logic_error("Invalid separation of figure placement string");

    std::array<std::pair<CIS::PieceType,PieceColor>,64> figurePlacement;
    std::fill(figurePlacement.begin(),figurePlacement.end(),std::make_pair(CIS::PieceType::unknown,PieceColor::empty));
        
    for(uint strPart=0; strPart<8; strPart++)
    {
        CIS::ChessRow row = static_cast<CIS::ChessRow>(7-strPart);
        std::string oneRowPieces = figurePlacementFENParts[strPart];
        //std::cout<<strPart<<":"<<oneRowPieces<<" ";
        uint colInt = 0;
        for(char c : oneRowPieces)
        {
            if(allowedCharsInRowPieces.find(c)==allowedCharsInRowPieces.end())
                throw std::logic_error("Invalid character in row pieces string");
            //std::cout<<"   1"<<std::endl;
            if(c>='1' && c<='8')
                colInt += c-'0';
            else if(c==' ')
            {
                CIS::ChessColumn col = static_cast<CIS::ChessColumn>(colInt);
                std::uint8_t index = CIS::squareToBoardIndex(CIS::Square(col,row));
                figurePlacement[index] = {CIS::PieceType::empty,PieceColor::empty};
                colInt += 1;
            }
            else
            {
                CIS::ChessColumn col = static_cast<CIS::ChessColumn>(colInt);
                CIS::Square sq(col,row);
                std::uint8_t index = CIS::squareToBoardIndex(sq);
                //std::cout<<c<<":"<<sq.to_string()<<"|"<<int(index)<<" ";
                PieceColor thisPieceColor;
                if(c>=97)
                {
                    c -= 32;
                    thisPieceColor = PieceColor::black;
                }
                else
                {
                    thisPieceColor = PieceColor::white;
                }
                
                CIS::PieceType thisPieceType;
                switch (c)
                {
                    case 'P':
                        thisPieceType = CIS::PieceType::pawn;
                        break;
                    case 'R':
                        thisPieceType = CIS::PieceType::rook;
                        break;
                    case 'N':
                        thisPieceType = CIS::PieceType::knight;
                        break;
                    case 'B':
                        thisPieceType = CIS::PieceType::bishop;
                        break;
                    case 'Q':
                        thisPieceType = CIS::PieceType::queen;
                        break;
                    case 'K':
                        thisPieceType = CIS::PieceType::king;
                        break;
                    default:
                        throw std::logic_error("Invalid character as type");
                }
                figurePlacement[index] = {thisPieceType,thisPieceColor};
                colInt += 1;
            }
        }
        //std::cout<<std::endl;
        //std::cout<<strPart<<":|"<<oneRowPieces<<"| -- "<<colInt<<std::endl;;
        if(colInt!=8)
        {
            throw std::logic_error("Sum of one row does not match the required 8");
        }
    }
    
    //throw std::runtime_error("Temp Stop");
    return figurePlacement;
}

std::unique_ptr<RBCAgent::FullChessInfo> RBCAgent::decodeObservation
(
    StateObj *pos,
    PieceColor observer,
    PieceColor observationTarget
) const
{
    using CIS = ChessInformationSet;
    open_spiel::chess::Color os_observer = AgentColor_to_OpenSpielColor(observer);
    open_spiel::chess::Color os_observationTarget = AgentColor_to_OpenSpielColor(observationTarget);
    
    //std::cout<<"Decode state via state string"<<std::endl;
    /*
     * Decode state via state string
     */
    std::string observationString = pos->get_state_string(os_observer,os_observationTarget);
    //std::cout<<"Got string"<<std::endl;
    //uint observationStringLen = observationString.size();
    std::vector<std::string> observationStringParts;
    splitObsFEN(observationString,observationStringParts);
    std::string figurePlacementString = observationStringParts[0];
    std::string castlingString = observationStringParts[1];
    std::string phaseString = observationStringParts[2];
    std::string captureString = observationStringParts[3];
    std::string sideToPlayString = observationStringParts[4];
    std::string illegalMoveString = observationStringParts[5];
    /*
    std::cout<<"figurePlacementString:"<<figurePlacementString<<"|"<<std::endl;
    std::cout<<"castlingString:"<<castlingString<<"|"<<std::endl;
    std::cout<<"phaseString:"<<phaseString<<"|"<<std::endl;
    std::cout<<"captureString:"<<captureString<<"|"<<std::endl;
    std::cout<<"sideToPlayString:"<<sideToPlayString<<"|"<<std::endl;
    std::cout<<"illegalMoveString:"<<illegalMoveString<<"|"<<std::endl;
    */
    std::array<std::pair<ChessInformationSet::PieceType,RBCAgent::PieceColor>,64> figureBoard = FullChessInfo::decodeFENFigurePlacement(figurePlacementString);
    /*
    auto pieceTypeName = [](std::pair<CIS::PieceType,PieceColor> stringDat)
    {
        if(stringDat.first==CIS::PieceType::empty ||
           stringDat.first==CIS::PieceType::unknown ||
           stringDat.second==PieceColor::empty)
            return "";
        std::string pieceTypeName;
        if(stringDat.second==PieceColor::white)
            referencePieceTypeName.append("white");
        else
            referencePieceTypeName.append("black");
        switch (stringDat.first)
        {
            case CIS::PieceType::pawn:
                referencePieceTypeName.append(" pawns");
                break;
            case CIS::PieceType::knight:
                referencePieceTypeName.append(" knights");
                break;
            case CIS::PieceType::bishop:
                referencePieceTypeName.append(" bishops");
                break;
            case CIS::PieceType::rook:
                referencePieceTypeName.append(" rooks");
                break;
            case CIS::PieceType::queen:
                referencePieceTypeName.append(" queens");
                break;
            case CIS::PieceType::king:
                referencePieceTypeName.append(" kings");
                break;
            default:
                throw std::logic_error("False type");
        }
        std::cout<<"|"<<pieceTypeName<<" == "<<referencePieceTypeName<<"|"<<std::endl;
        return pieceTypeName==referencePieceTypeName;
    };
    */
    
    //std::cout<<"Decode state via state tensor"<<std::endl;
    /*
     * Decode state via state tensor
     */
    
    // Get state plane tensor
    std::vector<float> statePlane;
    pos->get_state_planes(true,statePlane,1,os_observer,os_observationTarget);
    if(net->get_batch_size() * net->get_nb_input_values_total() != statePlane.size())
        throw std::logic_error("Incompatible sizes");
    //std::cout<<"Got tensor"<<std::endl;
    
    std::uint16_t offset = 0;    
    auto info = std::make_unique<FullChessInfo>();
    std::array<CIS::OnePlayerChessInfo*,2> obs = {&(info->white),&(info->black)};
    
    auto printPlane = [&](std::uint16_t offset, std::string planeName)
    {
        std::cout<<"Plane "<<planeName<<std::endl;
        std::string planeStr;
        planeStr=planeStr+"    a  b  c  d  e  f  g  h \n";
        planeStr=planeStr+"    ---------------------- \n";
        for(int8_t rank=7;rank>=0;rank--)
        {
            planeStr=planeStr+std::to_string(rank+1)+" |";
            for(int8_t file=0;file<8;file++)
            {
                const size_t index = open_spiel::chess::SquareToIndex(open_spiel::chess::Square{file, rank}, 8);
                planeStr = planeStr + " " + std::to_string((bool)statePlane[offset+index]) + " ";
            }
            planeStr=planeStr+"\n";
        }
        planeStr=planeStr+"\n";
        std::cout<<planeStr<<std::endl;
    };

    // Decoding plane functions
    auto pieceReader = [&](std::vector<CIS::Square>& piecesList, uint limit, std::string pieceTypeName)
    {
        for(unsigned int index=0; index<64; index++)
        {
            // Observation Tensor not used for piece deduction because sensing window not implemented in openspiel
            if(statePlane[index+offset]>0.5)
            {
                piecesList.push_back(CIS::boardIndexToSquare(index));
            }
        }
        if(piecesList.size()>limit)
            throw std::logic_error("Can not have more than "+std::to_string(limit)+" of "+pieceTypeName);
        offset+=64;
    };
    auto scalarReader = [&]()
    {
        float scalar = statePlane[0+offset];
        offset+=64;
        return scalar;
    };
    auto binaryReader = [&]()
    {
        float num = scalarReader();
        return (num==0.0f)?false:true;
    };
    
    MovePhase currentPhase;
    if(phaseString=="s")
        currentPhase = MovePhase::Sense;
    else if(phaseString=="m")
        currentPhase = MovePhase::Move;
    else
        throw std::logic_error("Illegal phase string: Must be (s,m)");
    info->currentPhase = currentPhase;
       
    bool lastMoveCapturedPiece=false;
    if(captureString=="c")
        lastMoveCapturedPiece = true;
    else if(captureString=="-")
        lastMoveCapturedPiece = false;
    else
        throw std::logic_error("Illegal capture string: Must be (c,-)");
    info->lastMoveCapturedPiece = lastMoveCapturedPiece;    
    
    PieceColor currentSideToPlay = PieceColor::empty;
    if(sideToPlayString=="w")
        currentSideToPlay = PieceColor::white;
    else if(sideToPlayString=="b")
        currentSideToPlay = PieceColor::black;
    else
        throw std::logic_error("Illegal sideToPlay string: Must be (w,b)");
    info->nextTurn = currentSideToPlay;
    
    bool lastMoveIllegal=false;
    if(illegalMoveString=="c")
        lastMoveIllegal = true;
    else if(illegalMoveString=="-")
        lastMoveIllegal = false;
    else
        throw std::logic_error("Illegal move string: Must be (c,-)");
    info->lastMoveIllegal = lastMoveIllegal;
    
    //pieces position white & black 0-11
    for(std::uint8_t index=0; index<figureBoard.size(); index++)
    {
        const std::pair<CIS::PieceType,RBCAgent::PieceColor>& squarePiece = figureBoard[index];
        if(squarePiece.second != PieceColor::empty)
        {
            std::uint8_t colorInd = static_cast<std::uint8_t>(squarePiece.second);
            switch (squarePiece.first)
            {
                case CIS::PieceType::pawn:
                    obs[colorInd]->pawns.push_back(CIS::boardIndexToSquare(index));
                    break;
                case CIS::PieceType::knight:
                    obs[colorInd]->knights.push_back(CIS::boardIndexToSquare(index));
                    break;
                case CIS::PieceType::bishop:
                    obs[colorInd]->bishops.push_back(CIS::boardIndexToSquare(index));
                    break;
                case CIS::PieceType::rook:
                    obs[colorInd]->rooks.push_back(CIS::boardIndexToSquare(index));
                    break;
                case CIS::PieceType::queen:
                    obs[colorInd]->queens.push_back(CIS::boardIndexToSquare(index));
                    break;
                case CIS::PieceType::king:
                    obs[colorInd]->kings.push_back(CIS::boardIndexToSquare(index));
                    break;
                default:
                    throw std::logic_error("Colored square must be a valid piece");
            }
        }
    }
    offset+=64*12;
    
    // repetitions 1&2 12-13
    uint nr_rep=0;
    bool repetitions_1 = binaryReader();
    bool repetitions_2 = binaryReader();
    if(repetitions_1==false && repetitions_2==false)
        nr_rep = 1;
    else if(repetitions_1==true && repetitions_2==false)
        nr_rep = 2;
    else if(repetitions_1==true && repetitions_2==true)
        nr_rep = 2;
    else
        throw std::logic_error("Invalid repetitions");
    
    // En_passant 14 (Not read from observation due to problems in rbc.cc : WriteTensor)
    std::vector<CIS::Square> en_passant;
    pieceReader(en_passant,1,"en_passant");
    /*
    obs[0]->en_passant=en_passant;
    obs[1]->en_passant=en_passant;
    */
    
    // Castling 15-18
    bool rightCastlingStr = false;
    bool leftCastlingStr = false;
    for(char letter : castlingString)
    {
        if(letter=='K')
            rightCastlingStr=true;
        else if(letter=='Q')
            leftCastlingStr=true;
        else if(letter!='-')
            throw std::logic_error("Invalid castling substring");
    }    
    for(std::uint16_t color=0; color<obs.size(); color++)
    {
        bool right_castling = binaryReader();
        obs[color]->kingside = right_castling;
        bool left_castling = binaryReader();
        obs[color]->queenside = left_castling;
    }
    std::int8_t colorInd = static_cast<std::int8_t>(selfColor);
    if(!(obs[colorInd]->kingside==rightCastlingStr &&
         obs[colorInd]->queenside==leftCastlingStr))
    {
        std::int8_t oppoColorInd = static_cast<std::int8_t>(PieceColor::black);
        std::cout<<"obs["<<int(oppoColorInd)<<"]->kingside:"<<obs[oppoColorInd]->kingside<<std::endl;
        std::cout<<"obs["<<int(oppoColorInd)<<"]->queenside:"<<obs[oppoColorInd]->queenside<<std::endl;
        std::cout<<"obs["<<int(colorInd)<<"]->kingside:"<<obs[colorInd]->kingside<<std::endl;
        std::cout<<"obs["<<int(colorInd)<<"]->queenside:"<<obs[colorInd]->queenside<<std::endl;
        std::cout<<"rightCastlingStr:"<<rightCastlingStr<<std::endl;
        std::cout<<"leftCastlingStr:"<<leftCastlingStr<<std::endl;
        throw std::logic_error("Castling mismatch or wrong color to play infered!");
    }

    // no_progress_count 19
    float no_progress_float = scalarReader();
    uint no_progress_count = static_cast<uint>(no_progress_float);
    obs[0]->no_progress_count=static_cast<std::uint8_t>(no_progress_count);
    obs[1]->no_progress_count=static_cast<std::uint8_t>(no_progress_count);
    
    offset += 16*64; //Last move
    offset += 1*64; //960 chess
    offset += 1*64; //White Piece Mask
    offset += 1*64; //Black Piece Mask
    offset += 1*64; //Checkerboard
    
    int whitePawnsExcess = static_cast<int>(scalarReader());
    int whiteKnightsExcess = static_cast<int>(scalarReader());
    int whiteBishopsExcess = static_cast<int>(scalarReader());
    int whiteRooksExcess = static_cast<int>(scalarReader());
    int whiteQueensExcess = static_cast<int>(scalarReader());

    offset += 1*64; //OP Bishops
    offset += 1*64; //Checkers
    
    int whitePawnsNumber = static_cast<int>(scalarReader());
    int whiteKnightsNumber = static_cast<int>(scalarReader());
    int whiteBishopsNumber = static_cast<int>(scalarReader());
    int whiteRooksNumber = static_cast<int>(scalarReader());
    int whiteQueensNumber = static_cast<int>(scalarReader());

    return info;
}

std::unique_ptr<std::vector<float>> RBCAgent::encodeStatePlane
(
    const std::unique_ptr<RBCAgent::FullChessInfo> fullState,
    const RBCAgent::PieceColor nextTurn,
    const unsigned int nextCompleteTurn
) const
{
    using CIS = ChessInformationSet;
    auto fullChessInfoPlane = std::make_unique<std::vector<float>>(net->get_nb_input_values_total(),0.0);
    std::vector<float>& infoPlane = *fullChessInfoPlane;
    std::array<CIS::OnePlayerChessInfo*,2> state = {&(fullState->white),&(fullState->black)};
    
    std::uint16_t offset = 0;
    for(std::uint16_t color=0; color<state.size(); color++)
    {
        if(state[color]->pawns.size()>8)
            std::logic_error("Can not have more than 8 pawns");
        for(const CIS::Square& sq : state[color]->pawns)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            infoPlane[offset+index] = 1.0;
        }
        offset+=64;
        
        if(state[color]->knights.size()>2)
            std::logic_error("Can not have more than 2 knights");
        for(const CIS::Square& sq : state[color]->knights)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            infoPlane[offset+index] = 1.0;
        }
        offset+=64;
        
        if(state[color]->bishops.size()>2)
            std::logic_error("Can not have more than 2 bishops");
        for(const CIS::Square& sq : state[color]->bishops)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            infoPlane[offset+index] = 1.0;
        }
        offset+=64;
        
        if(state[color]->rooks.size()>2)
            std::logic_error("Can not have more than 2 rooks");
        for(const CIS::Square& sq : state[color]->rooks)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            infoPlane[offset+index] = 1.0;
        }
        offset+=64;
        
        if(state[color]->queens.size()>9)
            std::logic_error("Can not have more than 9 queens");
        for(const CIS::Square& sq : state[color]->queens)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            infoPlane[offset+index] = 1.0;
        }
        offset+=64;
        
        if(state[color]->kings.size()>9)
            std::logic_error("Can not have more than 1 kings");
        for(const CIS::Square& sq : state[color]->kings)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            infoPlane[offset+index] = 1.0;
        }
        offset+=64;
    }
      
    auto putScalarToBoard = [](std::vector<float>& infoPlane, std::uint16_t offset, float value)
    {
        for(std::uint16_t index=offset; index<64; index++)
        {
            infoPlane[index] = value;
        }
    };
    
    float repetitions_1 = 0;
    putScalarToBoard(infoPlane,offset,repetitions_1);
    offset+=64;
    
    float repetitions_2 = 0;
    putScalarToBoard(infoPlane,offset,repetitions_2);
    offset+=64;
    
    std::array<float,5> pocketCountWhite = {0,0,0,0,0};
    for(float pocketCountPiece : pocketCountWhite)
    {
        putScalarToBoard(infoPlane,offset,pocketCountPiece);
        offset+=64;
    }
    
    std::array<float,5> pocketCountBlack = {0,0,0,0,0};
    for(float pocketCountPiece : pocketCountBlack)
    {
        putScalarToBoard(infoPlane,offset,pocketCountPiece);
        offset+=64;
    }
    
    float whitePromotions = 0;
    putScalarToBoard(infoPlane,offset,whitePromotions);
    offset+=64;
    
    float blackPromotions = 0;
    putScalarToBoard(infoPlane,offset,blackPromotions);
    offset+=64;
    
    if(nextTurn == PieceColor::white)
    {
        if(state[0]->en_passant_valid)
        {
            unsigned int index = CIS::squareToBoardIndex(state[0]->en_passant);
            infoPlane[offset+index] = 1.0;
        }
    }
    else if(nextTurn == PieceColor::black)
    {
        if(state[1]->en_passant_valid)
        {
            unsigned int index = CIS::squareToBoardIndex(state[1]->en_passant);
            infoPlane[offset+index] = 1.0;
        }
    }
    else
        throw std::logic_error("Current turn can not be empty");
    offset+=64;
    
    float colorVal = (nextTurn==PieceColor::white)?1.0:0.0;
    putScalarToBoard(infoPlane,offset,colorVal);
    offset+=64;
    
    putScalarToBoard(infoPlane,offset,nextCompleteTurn);
    offset+=64;
    
    for(std::uint16_t color=0; color<state.size(); color++)
    {
        if(state[color]->kingside)
        {
            putScalarToBoard(infoPlane,offset,1);
            offset+=64;
        }
        if(state[color]->queenside)
        {
            putScalarToBoard(infoPlane,offset,1);
            offset+=64;
        }
    }
    
    float halfTurns = std::min(fullState->white.no_progress_count,fullState->black.no_progress_count);
    putScalarToBoard(infoPlane,offset,halfTurns);
    offset+=64;

    return fullChessInfoPlane;
}

void RBCAgent::handleOpponentMoveInfo
(
    StateObj *pos
)
{
    std::cout<<"Player "<<selfColor<<" handles move information of opponent"<<std::endl;
    using CIS = ChessInformationSet;
    
    PieceColor oppoColor;
    if(selfColor==PieceColor::white)
        oppoColor=PieceColor::black;
    else if(selfColor==PieceColor::black)
        oppoColor=PieceColor::white;
    else
        throw std::logic_error("Invalid self Color");
    
    std::unique_ptr<FullChessInfo> observation = decodeObservation(pos,selfColor,oppoColor);
    CIS::OnePlayerChessInfo& selfObs = (selfColor==white)?observation->white:observation->black;
    CIS::OnePlayerChessInfo& selfState = playerPiecesTracker;

    bool onePieceCaptured = false;
    std::vector<CIS::BoardClause> conditions;
    
    // test for captured pawns
    auto pawnHere = selfObs.getBlockCheck(selfObs.pawns,CIS::PieceType::pawn);
    for(CIS::Square& sq : selfState.pawns)
    {
        if(!pawnHere(sq))
        // Pawn of self was captured
        {
            std::cout<<"Pawn captured at: "<<sq.to_string()<<std::endl;
            onePieceCaptured = true;
            bool inBoard;
            CIS::Square en_passant_sq = sq;
            if(selfColor==white)
                inBoard = en_passant_sq.vertMinus(1);
            else
                inBoard = en_passant_sq.vertPlus(1);
            if(!inBoard)
                throw std::logic_error("En-passant field can not be outside the field!");
            CIS::BoardClause capturedDirect(sq,CIS::BoardClause::PieceType::any);
            CIS::BoardClause capturedEnPassant(en_passant_sq,CIS::BoardClause::PieceType::pawn);
            if(playerPiecesTracker.en_passant_valid && playerPiecesTracker.en_passant==en_passant_sq)
            {
                conditions.push_back(capturedDirect | capturedEnPassant);
            }
            else
            {
                conditions.push_back(capturedDirect);
            }
        }
    }
    
    // test for all other captured pieces
    std::vector<std::tuple<std::vector<CIS::Square>*,CIS::PieceType,std::vector<CIS::Square>*>>   nonPawnPiecesList =
        {
         {&(selfObs.knights),CIS::PieceType::knight,&(selfState.knights)},
         {&(selfObs.bishops),CIS::PieceType::bishop,&(selfState.bishops)},
         {&(selfObs.rooks),  CIS::PieceType::rook,  &(selfState.rooks)},
         {&(selfObs.queens), CIS::PieceType::queen, &(selfState.queens)},
         {&(selfObs.kings),  CIS::PieceType::king,  &(selfState.kings)}
        };
    for(auto pieceTypeData : nonPawnPiecesList)
    {
        std::vector<CIS::Square>* selfObsPieceType = std::get<0>(pieceTypeData);
        CIS::PieceType pT = std::get<1>(pieceTypeData);
        std::vector<CIS::Square>* selfStatePieceType = std::get<2>(pieceTypeData);
        
        auto pieceTypeHere = selfObs.getBlockCheck(*selfObsPieceType,pT);
        for(const CIS::Square& sq : *selfStatePieceType)
        {
            if(!pieceTypeHere(sq))
            {
                if(onePieceCaptured)
                    throw std::logic_error("Multiple pieces can not be captured in one turn!");
                onePieceCaptured = true;
                CIS::BoardClause capturedDirect(sq,CIS::BoardClause::PieceType::any);
                conditions.push_back(capturedDirect);
            }
        }
    }
    
    cis->markIncompatibleBoards(conditions);
    cis->removeIncompatibleBoards();
}

void RBCAgent::handleSelfMoveInfo
(
    StateObj* rbcState
)
{
    using CIS = ChessInformationSet;
    
    if(OpenSpielColor_to_RBCColor(rbcState->currentPlayer()) != selfColor)
        throw std::logic_error("RBC game state is in the wrong current player status");
    if(OpenSpielPhase_to_RBCPhase(rbcState->currentPhase()) != MovePhase::Move)
        throw std::logic_error("RBC game state is in the wrong current phase");
    
// Decode action
    Action selfLastAction = this->evalInfo->bestMove;
    open_spiel::chess::Color os_self = AgentColor_to_OpenSpielColor(selfColor);
    std::cout<<"Player "<<selfColor<<" handles own move information: "<<rbcState->ActionToMoveString(selfLastAction,os_self)<<std::endl;
    std::tuple<std::uint8_t,open_spiel::chess::Move,bool> move = rbcState->ActionToIncompleteMove(selfLastAction,os_self);
    MovePhase actionPhaseType = static_cast<MovePhase>(std::get<0>(move));
    open_spiel::chess::Move& moveData = std::get<1>(move);
    bool passMove = std::get<2>(move);
    if(actionPhaseType == MovePhase::Sense)
    {
        throw std::logic_error("Invalid phase action");
    }
    if(!passMove)
    // No information can be infered from pass move
    {
        completeMoveData(moveData,playerPiecesTracker);
        open_spiel::chess::Move& selfLastMove = moveData;
        //std::cout<<"selfLastMove:"<<selfLastMove<<std::endl;
            
    //Apply action to state
        rbcState->do_action(evalInfo->bestMove);
        //std::cout<<"Action applied"<<std::endl;

    //Decode an evaluate after action status and infer information from that
        std::unique_ptr<FullChessInfo> observation = decodeObservation(rbcState,selfColor,selfColor);
        if(observation->nextTurn==selfColor || selfColor==PieceColor::empty)
            throw std::logic_error("Wrong turn marker");
        if(observation->currentPhase!=MovePhase::Sense)
            throw std::logic_error("Wrong move phase marker:"+std::to_string(observation->currentPhase));
                
        CIS::OnePlayerChessInfo& selfObs = (selfColor==white)?observation->white:observation->black;
        //std::function<std::pair<bool,CIS::PieceType>(const CIS::Square&)> squareToPiece;
        //squareToPiece = selfObs.getSquarePieceTypeCheck();
        
        //Test for castling
        bool castling = selfLastMove.is_castling;

        //Test for promotion
        CIS::PieceType promotionType = CIS::OpenSpielPieceType_to_CISPieceType(selfLastMove.promotion_type);
        bool promotion = (promotionType!=CIS::PieceType::empty)?true:false;        

        // Find moved piece and determine the squares
        CIS::Square fromSquare = CIS::Square(selfLastMove.from);
        //std::cout<<"fromSquare:"<<fromSquare.to_string()<<std::endl;
        CIS::Square toSquareAim = CIS::Square(selfLastMove.to);
        //std::cout<<"toSquareAim:"<<toSquareAim.to_string()<<std::endl;
        
        //std::cout<<"selfObs:"<<selfObs.to_string()<<std::endl;
        
        std::function<std::pair<bool,CIS::PieceType>(const CIS::Square&)> squareToPiecePreAction;
        squareToPiecePreAction = playerPiecesTracker.getSquarePieceTypeCheck();
        std::pair<bool,CIS::PieceType> fromPiece = squareToPiecePreAction(fromSquare);
        if(!fromPiece.first)
            throw std::logic_error("Move from empty square");
        CIS::PieceType initialMovePiece = fromPiece.second;
        //std::cout<<"MovePiece:"<<(int)initialMovePiece<<std::endl;

        //Find toSquareReal
        CIS::Square toSquareReal;
        std::vector<CIS::Square> previousStateMovePieces;
        std::vector<CIS::Square> currentStateMovePieces;    
        switch (initialMovePiece)
        {
            case CIS::PieceType::pawn:
                previousStateMovePieces = playerPiecesTracker.pawns;
                currentStateMovePieces = selfObs.pawns;
                break;
            case CIS::PieceType::knight:
                previousStateMovePieces = playerPiecesTracker.knights;
                currentStateMovePieces = selfObs.knights;
                break;
            case CIS::PieceType::bishop:
                previousStateMovePieces = playerPiecesTracker.bishops;
                currentStateMovePieces = selfObs.bishops;
                break;
            case CIS::PieceType::rook:
                previousStateMovePieces = playerPiecesTracker.rooks;
                currentStateMovePieces = selfObs.rooks;
                break;
            case CIS::PieceType::queen:
                previousStateMovePieces = playerPiecesTracker.queens;
                currentStateMovePieces = selfObs.queens;
                break;
            case CIS::PieceType::king:
                previousStateMovePieces = playerPiecesTracker.kings;
                currentStateMovePieces = selfObs.kings;
                break;
            default:
                throw std::logic_error("Moved piece is empty!");
        }
        if(promotion)
        {
            toSquareReal = toSquareAim;
        }
        else
        {
            std::unordered_set<CIS::Square,CIS::Square::Hasher> previousStateSquareSet(previousStateMovePieces.begin(),previousStateMovePieces.end());
            bool pieceMoved = false;
            for(const CIS::Square& sq : currentStateMovePieces)
            {
                auto iter = previousStateSquareSet.find(sq);
                if(iter==previousStateSquareSet.end())
                {
                    //std::cout<<"Moved:"<<sq.to_string()<<std::endl;
                    if(pieceMoved)
                    {
                        std::cout<<"Prev:";
                        for(auto sq : previousStateMovePieces)
                            std::cout<<" "<<sq.to_string();
                        std::cout<<std::endl<<"Curr:";
                        for(auto sq : currentStateMovePieces)
                            std::cout<<" "<<sq.to_string();
                        std::cout<<std::endl;
                        throw std::logic_error("More than one piece of one type");
                    }
                    pieceMoved=true;
                    toSquareReal = sq;
                }
            }
            if(!pieceMoved)
                toSquareReal = fromSquare;
        }
        //std::cout<<"toSquareReal:"<<toSquareReal.to_string()<<std::endl;
        

        std::vector<CIS::BoardClause> conditions;
        if(observation->lastMoveIllegal)
        {
            if(promotion)
                throw std::logic_error("Promotion can not be an illegal move");
            if(observation->lastMoveCapturedPiece)
            // piece movement stopped prematurely, special moves like en_passant, castling and promotion are not possible here
            {
                if(fromSquare==toSquareReal)
                    throw std::logic_error("Capture while no piece movement!");
                if(castling)
                    throw std::logic_error("Castling can not capture piece!");
                CIS::BoardClause pieceAtToSquare(toSquareReal,CIS::BoardClause::PieceType::any);
                conditions.push_back(pieceAtToSquare);
            }
            else
            {
                if(fromSquare!=toSquareReal)
                    throw std::logic_error("Illegal move but movement and no capture!");
                if(castling)
                // enemy piece prevents castling
                {
                    CIS::ChessRow castlingRow = (selfColor==PieceColor::black)?CIS::ChessRow::eight:CIS::ChessRow::one;
                    std::vector<CIS::ChessColumn> castlingCol;
                    if(playerPiecesTracker.kingside && !selfObs.kingside)
                        //move was castling to kingside
                        castlingCol = {CIS::ChessColumn::F,CIS::ChessColumn::G};
                    else if(playerPiecesTracker.queenside && !selfObs.queenside)
                        //mode was castling to queenside
                        castlingCol = {CIS::ChessColumn::B,CIS::ChessColumn::C,CIS::ChessColumn::D};
                    else
                        throw std::logic_error("Castling move but no castling rights!");
                    
                    CIS::BoardClause castlingClause;
                    for(CIS::ChessColumn col : castlingCol)
                    {
                        CIS::Square sq(col,castlingRow);
                        castlingClause = castlingClause | CIS::BoardClause(sq,CIS::BoardClause::PieceType::none);
                    }
                    conditions.push_back(castlingClause);
                }
                else if(initialMovePiece==CIS::PieceType::pawn)
                // enemy piece prevents pawn movement
                {
                    if(fromSquare.column==toSquareAim.column)
                    // Failed forward move
                    {
                        CIS::Square squareBeforePawn = fromSquare;
                        if(selfColor==PieceColor::white)
                            squareBeforePawn.vertPlus(1);
                        else
                            squareBeforePawn.vertMinus(1);
                        conditions.push_back(CIS::BoardClause(squareBeforePawn,CIS::BoardClause::PieceType::any));
                    }
                    else
                    // Failed en_passant move
                    {
                        CIS::Square squareNextToPawn = {toSquareAim.column,fromSquare.row};
                        CIS::BoardClause noPawnHere(squareNextToPawn,CIS::BoardClause::PieceType::pawn);
                        noPawnHere = !noPawnHere;
                        conditions.push_back(noPawnHere);
                    }
                }
                else
                {
                    throw std::logic_error("Non castling and non pawn illegal move must be capturing!");
                }
            }
        }
        else
        {
            if(toSquareAim!=toSquareReal)
                throw std::logic_error("Legal move but target and reality differs!");
            if(observation->lastMoveCapturedPiece)
            {
                if(fromSquare==toSquareReal)
                    throw std::logic_error("Capture while no piece movement!");
                if(castling)
                    throw std::logic_error("Castling can not capture piece!");
                
                if(initialMovePiece==CIS::PieceType::pawn)
                // pawn captures piece in two possible ways
                {
                    CIS::Square squareNextToPawn = {toSquareAim.column,fromSquare.row};
                    CIS::BoardClause en_passantCapture(squareNextToPawn,CIS::BoardClause::PieceType::pawn);
                    CIS::BoardClause conventionalCapture(toSquareReal,CIS::BoardClause::PieceType::any);
                    conditions.push_back(en_passantCapture | conventionalCapture);
                }
                else
                //capture by any other piece
                {
                    conditions.push_back(CIS::BoardClause(toSquareReal,CIS::BoardClause::PieceType::any));
                }
            }
            else
            {
                if(initialMovePiece==CIS::PieceType::knight)
                {
                    conditions.push_back(CIS::BoardClause(toSquareReal,CIS::BoardClause::PieceType::none));
                }
                else
                {
                    auto [delta_col,delta_row] = fromSquare.diffToSquare(toSquareReal);
                    
                    enum Dir {Straight,Diagonal};
                    Dir movementDir;
                    if(fromSquare.row==toSquareReal.row || fromSquare.column==toSquareReal.column)
                    {
                        movementDir = Dir::Straight;
                        if(delta_col!=0 && delta_row!=0)
                            throw std::logic_error("Invalid movement difference!");
                    }
                    else
                    {
                        movementDir = Dir::Diagonal;
                        if(std::abs(delta_col)!=std::abs(delta_row))
                            throw std::logic_error("Invalid movement difference!");
                    }
                    delta_col = (delta_col!=0)?delta_col/std::abs(delta_col):0;
                    delta_row = (delta_row!=0)?delta_row/std::abs(delta_row):0;
                    
                    CIS::Square squareIteration = fromSquare;
                    while(squareIteration!=toSquareReal)
                    {
                        bool valid = squareIteration.moveSquare(delta_col,delta_row);
                        if(!valid)
                            throw std::logic_error("Piece can not move over invalid square");
                        conditions.push_back(CIS::BoardClause(squareIteration,CIS::BoardClause::PieceType::none));
                    }
                }
            }
        }
            
        cis->markIncompatibleBoards(conditions);
        cis->removeIncompatibleBoards();
        
    //Undo action to state
        rbcState->undo_action(evalInfo->bestMove);
        
    // Advance own state using move
        playerPiecesTracker.applyMove(fromSquare,toSquareReal,initialMovePiece,promotionType,castling);
    }
}

void RBCAgent::handleScanInfo
(
    StateObj *pos,
    ChessInformationSet::Square scanCenter
)
{
    using CIS = ChessInformationSet;
    
    std::cout<<"Player "<<selfColor<<" handles scan information at square "<<scanCenter.to_string()<<" -- "<<int(CIS::squareToBoardIndex(scanCenter))<<std::endl;
    
    PieceColor oppoColor;
    if(selfColor==PieceColor::white)
        oppoColor=PieceColor::black;
    else if(selfColor==PieceColor::black)
        oppoColor=PieceColor::white;
    else
        throw std::logic_error("Invalid self Color");
    
    std::unique_ptr<FullChessInfo> observation = decodeObservation(pos,selfColor,oppoColor);
    CIS::OnePlayerChessInfo& opponentObs = (selfColor==white)?observation->black:observation->white;
    
    std::vector<CIS::BoardClause> conditions;
    
    std::vector<std::tuple<std::vector<CIS::Square>*,CIS::PieceType>> piecesList =
        {
         {&(opponentObs.pawns),  CIS::PieceType::pawn},
         {&(opponentObs.knights),CIS::PieceType::knight},
         {&(opponentObs.bishops),CIS::PieceType::bishop},
         {&(opponentObs.rooks),  CIS::PieceType::rook},
         {&(opponentObs.queens), CIS::PieceType::queen},
         {&(opponentObs.kings),  CIS::PieceType::king}
        };
    for(auto pieceTypeData : piecesList)
    {
        std::vector<CIS::Square>* opponentObsPieceType = std::get<0>(pieceTypeData);
        CIS::PieceType pT = std::get<1>(pieceTypeData);
        unsigned int pT_int = static_cast<unsigned int>(pT);
        CIS::BoardClause::PieceType pT_Clause = static_cast<CIS::BoardClause::PieceType>(pT_int);
        
        for(CIS::Square sq : *opponentObsPieceType)
        {
            CIS::BoardClause observedPiece(sq,pT_Clause);
            conditions.push_back(observedPiece);
        }
    }
    cis->markIncompatibleBoards(conditions);
    cis->removeIncompatibleBoards();
    
    std::cout<<"Scanning done"<<std::endl;
}

ChessInformationSet::Square RBCAgent::selectScanAction
(
    StateObj *pos
)
{
    using CIS=ChessInformationSet;
    
    unsigned short col = randomScanDist(gen);
    unsigned short row = randomScanDist(gen);
    CIS::Square randomScanSq = {static_cast<CIS::ChessColumn>(col),static_cast<CIS::ChessRow>(row)};
    //return randomScanSq;
    return CIS::Square{CIS::ChessColumn::F,CIS::ChessRow::six};
}

std::unique_ptr<RBCAgent::FullChessInfo> RBCAgent::selectHypothese()
{
    std::cout<<"Player "<<selfColor<<" selects one hypothese of the opponents state"<<std::endl;
    
    using CIS = ChessInformationSet;
    
    //std::cout<<"cis->validSize():"<<cis->validSize()<<std::endl;
    //std::cout<<"cis->size():"<<cis->size()<<std::endl;    

    randomHypotheseSelect = std::uniform_int_distribution<std::uint64_t>(0,cis->size()-1);
    std::uint64_t selectedBoard = randomHypotheseSelect(gen);
    //std::cout<<"Selected Board:"<<selectedBoard<<std::endl;
    auto iter = cis->begin();
    //std::cout<<"Created Iterator"<<std::endl;
    //std::cout<<"getCurrentIndex:"<<iter.getCurrentIndex()<<std::endl;
    for(int i=0;i<selectedBoard;i++)
    {
        if(iter==cis->end())
            iter = cis->begin();
        else
            iter++;
    }
    //std::cout<<"Enter iterator"<<std::endl;
    //std::cout<<"getCurrentIndex:"<<iter.getCurrentIndex()<<std::endl;
    if(iter == cis->end())
        throw std::logic_error("Chess information set is empty");
    std::unique_ptr<std::pair<CIS::OnePlayerChessInfo,double>> selectedHypothese = *iter;
    
    //std::cout<<"cis->size():"<<cis->size()<<std::endl;
    //std::cout<<"selectedBoard:"<<selectedBoard<<std::endl;
    
    auto fullInfoSet = std::make_unique<FullChessInfo>();
    
    //std::cout<<"selfColor:"<<selfColor<<std::endl;
    
    if(selfColor == PieceColor::white)
    {
        fullInfoSet->white = playerPiecesTracker;
        fullInfoSet->black = selectedHypothese->first;
        fullInfoSet->nextTurn = PieceColor::white;
    }
    else
    {
        fullInfoSet->black = playerPiecesTracker;
        fullInfoSet->white = selectedHypothese->first;
        fullInfoSet->nextTurn = PieceColor::black;
    }
    fullInfoSet->currentPhase = MovePhase::Move;
    fullInfoSet->lastMoveCapturedPiece = false;
    fullInfoSet->lastMoveIllegal = false;
    fullInfoSet->nextCompleteTurn = currentTurn;
    
    /*
    std::cout<<"fullInfoSet->black.kingside:"<<fullInfoSet->black.kingside<<std::endl;
    std::cout<<"fullInfoSet->black.queenside:"<<fullInfoSet->black.queenside<<std::endl;
    std::cout<<"fullInfoSet->white.kingside:"<<fullInfoSet->white.kingside<<std::endl;
    std::cout<<"fullInfoSet->white.queenside:"<<fullInfoSet->white.queenside<<std::endl;
    
    std::cout<<"fullInfoSet->nextTurn:"<<fullInfoSet->nextTurn<<std::endl;
    */
    std::cout<<"Selected as hypothese: "<<fullInfoSet->getFEN()<<std::endl;
    return fullInfoSet;
}

StateObj* RBCAgent::setupMoveSearchState()
{
    chessState = new OpenSpielState(open_spiel::gametype::SupportedOpenSpielVariants::CHESS);
    std::unique_ptr<FullChessInfo> searchState = selectHypothese();
    chessState->set(searchState->getFEN(),false,open_spiel::gametype::SupportedOpenSpielVariants::CHESS);
    return chessState;
}

ChessInformationSet::Square RBCAgent::applyScanAction
(
    StateObj *pos
)
{
    CIS::Square scanSq = selectScanAction(pos);
    pos->do_action(CIS::squareToBoardIndex(scanSq));
    return scanSq;
}

void RBCAgent::stepForwardHypotheses()
{
    std::cout<<"Player "<<selfColor<<" advances its hypotheses of the opponents state"<<std::endl;
    
    cis->clearRemoved();
    //std::cout<<"Clearing removed states of CIS"<<std::endl;
    auto newCis = std::make_unique<CIS>();
    //std::cout<<"Creating new instance of CIS"<<std::endl;
    
    if(cis->size()<1 || cis->size()!=cis->validSize())
        throw std::logic_error("Advancing hypotheses from invalid cis size");    
    
    std::vector<std::uint64_t> cisInds;
    for(auto iter=cis->begin(); iter!=cis->end(); iter++)
    {
        //std::cout<<"Iter:"<<std::endl;
        std::unique_ptr<std::pair<CIS::OnePlayerChessInfo,double>> oldHypothese = *iter;
        CIS::OnePlayerChessInfo& hypoPiecesOpponent = oldHypothese->first;
        double probability = (*iter)->second;
        std::unique_ptr<std::vector<std::pair<CIS::OnePlayerChessInfo,double>>> newHypotheses;
        newHypotheses = generateHypotheses(hypoPiecesOpponent);
        for(auto& oneHypothese : *newHypotheses)
            oneHypothese.second = probability;
        //std::cout<<"Developed: "<<iter.getCurrentIndex()<<std::endl;
        cisInds.push_back(iter.getCurrentIndex());
        //std::cout<<"Removed"<<std::endl;

        try
        {
            newCis->add(*newHypotheses);
        }
        catch(const std::bad_alloc& e)
        {
            cis->remove(cisInds);
            cis->clearRemoved();
            iter = cis->begin();
            newCis->add(*newHypotheses);
        }
    }
    cis = std::move(newCis);
    std::cout<<"Player "<<selfColor<<" hypotheses generation done"<<std::endl;
}
