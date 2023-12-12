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
    PlaySettings* playSettings
):
MCTSAgent(netSingle, netBatches, searchSettings, playSettings),
currentTurn(0)
{
    std::cout<<"Create RBCAgent"<<std::endl;
    gen = std::mt19937(rd());
    randomScanDist = std::uniform_int_distribution<unsigned short>(1,6);
    cis = std::make_unique<ChessInformationSet>();
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
            
        for(const CIS::Square& sq : oneColorInfo.en_passant)
        {
            if(enPassantString.size()>0)
                throw std::logic_error("More than one en_passant is not possible!");
            enPassantString += sq.to_string();
            /*
            unsigned int column = static_cast<unsigned int>(sq.column);
            unsigned int row = static_cast<unsigned int>(sq.row);
            char columnChar = column + 97;
            char rowChar = row + 60;
            enPassantString += columnChar;
            enPassantString += rowChar;
            */
        }
        
        charOffset+=32;
    }
    
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

void RBCAgent::set_search_settings
(
    StateObj *pos,
    SearchLimits *searchLimits,
    EvalInfo* evalInfo
)
{
    //Reduce hypotheses using the previous move information
    if(currentTurn!=0 || selfColor!=PieceColor::black)
    {
        handleOpponentMoveInfo(pos);
        stepForwardHypotheses();
    }
    
    //Scan the board an reduce hypotheses
    ChessInformationSet::Square scanCenter = applyScanAction(pos);
    handleScanInfo(pos,scanCenter);
    
    // setup MCTS search
    this->evalInfo = evalInfo;
    StateObj* chessOpenSpiel = setupMoveSearchState();
    MCTSAgent::set_search_settings(chessOpenSpiel,searchLimits,evalInfo);
}

void RBCAgent::perform_action()
{
    // Run mcts tree and set action to game
    MCTSAgent::perform_action();
    
    //Reduce hypotheses using the own move information
    state->do_action(evalInfo->bestMove);
    handleSelfMoveInfo(state);
    state->undo_action(evalInfo->bestMove);
    delete chessOpenSpiel;
}

std::unique_ptr<std::vector<std::pair<ChessInformationSet::OnePlayerChessInfo,double>>> RBCAgent::generateHypotheses
(
    ChessInformationSet::OnePlayerChessInfo& piecesOpponent,
    ChessInformationSet::OnePlayerChessInfo& piecesSelf,
    const RBCAgent::PieceColor selfColor
) const
{
    if(selfColor == PieceColor::empty)
        throw std::invalid_argument("selfColor must not be PieceColor::empty");
    
    using CIS_Square = ChessInformationSet::Square;
    using CIS_CPI = ChessInformationSet::OnePlayerChessInfo;
    auto hypotheses = std::make_unique<std::vector<std::pair<CIS_CPI,double>>>();
    
    FullChessInfo fullState;
    PieceColor opponentColor;
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
    OpenSpielState hypotheticState(open_spiel::gametype::SupportedOpenSpielVariants::RBC);
    hypotheticState.set(fen,false,open_spiel::gametype::SupportedOpenSpielVariants::RBC);
    std::vector<Action> legal_actions_int = hypotheticState.legal_actions();
    std::vector<open_spiel::chess::Move> legal_actions_move(legal_actions_int.size());
    for(unsigned int actionInd=0; actionInd<legal_actions_int.size(); actionInd++)
    {
        legal_actions_move[actionInd] = hypotheticState.ActionToMove(legal_actions_int[actionInd]);
    }
    for(const open_spiel::chess::Move& move : legal_actions_move)
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
        
        //process possible promotion of pawn
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
            
            std::function<std::vector<CIS::Square>::iterator(const CIS::Square&)> pawnGetter = new_hypothese.getPieceIter(new_hypothese.pawns);
            
            auto pawnIter = pawnGetter(from);
            if(pawnIter==new_hypothese.pawns.end())
                throw std::logic_error("Pawn moves from position where it does not sit!");
            
            new_hypothese.pawns.erase(pawnIter);
            switch(promPieceType)
            {
                case CIS::PieceType::knight:
                    new_hypothese.knights.push_back(to);
                    break;
                case CIS::PieceType::bishop:
                    new_hypothese.bishops.push_back(to);
                    break;
                case CIS::PieceType::rook:
                    new_hypothese.rooks.push_back(to);
                    break;
                case CIS::PieceType::queen:
                    new_hypothese.queens.push_back(to);
                    break;
                default:
                    throw std::logic_error("Piece promoted to non valid piece type!");
            }
            new_hypothese.no_progress_count=0;
            
            continue;
        }
        
        //process possible castling
        if(castling)
        {
           enum Castling {queenside,kingside};
            Castling side;
            
            if(pieceType!=CIS::PieceType::king)
                throw std::logic_error("Castling but king not moved!");            
            if(new_hypothese.kings.size()!=1)
                throw std::logic_error("There must be exactly one king!");
            CIS::Square& theKing = new_hypothese.kings[0];
            theKing = to;
            CIS::Square rookDest = to;
            
            if(from.column < to.column)
            {
                rookDest.horizMinus(1);
                side = Castling::kingside;
                if(!new_hypothese.kingside)
                    throw std::logic_error("Castling move kingside but illegal!");
            }
            else if(from.column > to.column)
            {
                rookDest.horizPlus(1);               
                side = Castling::queenside;
                if(!new_hypothese.queenside)
                    throw std::logic_error("Castling move queenside but illegal!");
            }
            else
                throw std::logic_error("No movement in castling!");
            
            std::function<std::vector<CIS::Square>::iterator(const CIS::Square&)> rookGetter = new_hypothese.getPieceIter(new_hypothese.rooks);
            
            auto rookIter = new_hypothese.rooks.end();
            if(side==Castling::kingside)
                rookIter = rookGetter({CIS::ChessColumn::H,theKing.row});
            else
                rookIter = rookGetter({CIS::ChessColumn::A,theKing.row});            
            if(rookIter==new_hypothese.rooks.end())
                throw std::logic_error("No rook on initial position in castling move!");

            *rookIter = rookDest;
            
            new_hypothese.kingside=false;
            new_hypothese.queenside=false;

            continue;
        }
        
        //process all other movement
        if(pieceType==CIS::PieceType::pawn)
        {
            std::function<std::vector<CIS::Square>::iterator(const CIS::Square&)> pawnGetter = new_hypothese.getPieceIter(new_hypothese.pawns);
            int step = static_cast<int>(to.row)-static_cast<int>(from.row);
            uint stepSize = std::abs(step);
            bool doubleStep = (stepSize==2)?true:false;
            if(stepSize!=1 && stepSize!=2)
                throw std::logic_error("Pawn must move eiter one or two steps forward!");
            auto pawnIter = pawnGetter(from);
            if(pawnIter==new_hypothese.pawns.end())
                throw std::logic_error("Moved pawn from nonexistant position!");
            *pawnIter = to;
            if(doubleStep)
            {
                CIS::Square en_passant_sq = from;
                (step<0)?en_passant_sq.vertMinus(1):en_passant_sq.vertPlus(1);
                new_hypothese.en_passant.push_back(en_passant_sq);
            }
        }
        else if(pieceType==CIS::PieceType::rook)
        {
            std::function<std::vector<CIS::Square>::iterator(const CIS::Square&)> rookGetter = new_hypothese.getPieceIter(new_hypothese.rooks);
            auto rookIter = rookGetter(from);
            if(rookIter==new_hypothese.rooks.end())
                throw std::logic_error("Moved rook from nonexistant position!");
            if(from.row==CIS::ChessRow::one || from.row==CIS::ChessRow::eight)
            {
                if(from.column==CIS::ChessColumn::A)
                    new_hypothese.queenside=false;
                if(from.column==CIS::ChessColumn::H)
                    new_hypothese.kingside=false;
            }
            *rookIter = to;
        }
        else if(pieceType==CIS::PieceType::knight)
        {
            std::function<std::vector<CIS::Square>::iterator(const CIS::Square&)> knightGetter = new_hypothese.getPieceIter(new_hypothese.knights);
            auto knightIter = knightGetter(from);
            if(knightIter==new_hypothese.knights.end())
                throw std::logic_error("Moved knight from nonexistant position!");
            *knightIter = to;
        }
        else if(pieceType==CIS::PieceType::bishop)
        {
            std::function<std::vector<CIS::Square>::iterator(const CIS::Square&)> bishopGetter = new_hypothese.getPieceIter(new_hypothese.bishops);
            auto bishopIter = bishopGetter(from);
            if(bishopIter==new_hypothese.bishops.end())
                throw std::logic_error("Moved bishop from nonexistant position!");
            *bishopIter = to;
        }
        else if(pieceType==CIS::PieceType::queen)
        {
            std::function<std::vector<CIS::Square>::iterator(const CIS::Square&)> queenGetter = new_hypothese.getPieceIter(new_hypothese.queens);
            auto queenIter = queenGetter(from);
            if(queenIter==new_hypothese.queens.end())
                throw std::logic_error("Moved queen from nonexistant position!");
            *queenIter = to; 
        }
        else if(pieceType==CIS::PieceType::king)
        {
            if(new_hypothese.kings.size()!=1)
                throw std::logic_error("There must be exactly one king!");
            CIS::Square& theKing = new_hypothese.kings[0];
            theKing = to;
            CIS::Square rookDest = to;
            new_hypothese.kingside=false;
            new_hypothese.queenside=false;
        }
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

std::unique_ptr<RBCAgent::FullChessInfo> RBCAgent::decodeObservation
(
    StateObj *pos
) const
{
    using CIS = ChessInformationSet;
    
    auto inputPlanesSmPtr = std::make_unique<float[]>(net->get_batch_size() * net->get_nb_input_values_total());   
    float* inputPlanes =  inputPlanesSmPtr.get();
    pos->get_state_planes(true,inputPlanes,1);
    
    std::string observationString = pos->get_state_string();
    std::vector<std::string> observationStringParts;
    std::string::size_type index;
    while((index=observationString.find(" "))!=std::string::npos)
    {
        observationStringParts.push_back(observationString.substr(0,index));
        observationString = observationString.substr(index+1);
    }
    observationStringParts.push_back(observationString);
    if(observationStringParts.size()!=6)
        throw std::logic_error("Invalid observation string");

    
    std::uint16_t offset = 0;    
    auto info = std::make_unique<FullChessInfo>();
    std::array<CIS::OnePlayerChessInfo*,2> obs = {&(info->white),&(info->black)};
    
    auto pieceReader = [&](std::vector<CIS::Square>& piecesList, uint limit, std::string pieceTypeName)
    {
        std::array<float,64> board;
        std::memcpy(board.data(),pos+offset,64);
        for(unsigned int index=0; index<piecesList.size(); index++)
        {
            if(board[index]>0.5)
            {
                piecesList.push_back(CIS::boardIndexToSquare(index));
            }
        }
        if(piecesList.size()>limit)
            std::logic_error("Can not have more than "+std::to_string(limit)+" of "+pieceTypeName);
        offset+=64;
    };
    
    auto scalarReader = [&]()
    {
        std::array<float,64> board;
        std::memcpy(board.data(),pos+offset,64);
        offset+=64;
        return board[0];
    };
    
    auto binaryReader = [&]()
    {
        float num = scalarReader();
        return (num==0.0f)?false:true;
    };
    
    std::string phaseString = observationStringParts[2];
    MovePhase currentPhase;
    if(phaseString=="s")
        currentPhase = MovePhase::sense;
    else if(phaseString=="m")
        currentPhase = MovePhase::move;
    else
        throw std::logic_error("Illegal phaseString string");
    info->currentPhase = currentPhase;
    
    std::string captureString = observationStringParts[3];
    bool lastMoveCapturedPiece=false;
    if(captureString=="c")
        lastMoveCapturedPiece = true;
    else if(captureString=="-")
        lastMoveCapturedPiece = false;
    else
        throw std::logic_error("Illegal capture string");
    info->lastMoveCapturedPiece = lastMoveCapturedPiece;    
    
    std::string sideToPlayString = observationStringParts[4];
    PieceColor currentSideToPlay = PieceColor::empty;
    if(sideToPlayString=="w")
        currentSideToPlay = PieceColor::white;
    else if(sideToPlayString=="b")
        currentSideToPlay = PieceColor::black;
    else
        throw std::logic_error("Illegal sideToPlay string");
    info->nextTurn = currentSideToPlay;    
    
    std::string illegalMoveString = observationStringParts[5];
    bool lastMoveIllegal=false;
    if(illegalMoveString=="c")
        lastMoveIllegal = true;
    else if(illegalMoveString=="-")
        lastMoveIllegal = false;
    else
        throw std::logic_error("Illegal illegal move string");
    info->lastMoveIllegal = lastMoveIllegal;
    
    //pieces position white & black 0-11
    std::string piecesString = observationStringParts[0];
    for(std::uint16_t color=0; color<obs.size(); color++)
    {
        pieceReader(obs[color]->kings,1,"kings");
        pieceReader(obs[color]->queens,9,"queens");
        pieceReader(obs[color]->rooks,10,"rooks");
        pieceReader(obs[color]->bishops,10,"bishops");
        pieceReader(obs[color]->knights,10,"knights");
        pieceReader(obs[color]->pawns,8,"pawns");
    }
    
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
    
    // En_passant 14
    std::vector<CIS::Square> en_passant;
    pieceReader(en_passant,1,"en_passant");
    obs[0]->en_passant=en_passant;
    obs[1]->en_passant=en_passant;
    
    // Castling 15-18
    std::string castlingString = observationStringParts[1];
    bool rightCastlingStr = false;
    bool leftCastlingStr = false;
    for(char letter : castlingString)
    {
        if(letter='K')
            rightCastlingStr=true;
        else if(letter='Q')
            leftCastlingStr=true;
        else
            throw std::logic_error("Invalid castling substring");
    }    
    for(std::uint16_t color=0; color<obs.size(); color++)
    {
        bool right_castling = binaryReader();
        obs[color]->kingside = right_castling;
        bool left_castling = binaryReader();
        obs[color]->queenside = left_castling;
    }
    std::int8_t colorInd = static_cast<std::int8_t>(currentSideToPlay);
    if(!(obs[colorInd]->kingside==rightCastlingStr &&
         obs[colorInd]->queenside==leftCastlingStr))
        throw std::logic_error("Castling mismatch or wrong color to play infered!");

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
        for(const CIS::Square& sq : state[0]->en_passant)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            infoPlane[offset+index] = 1.0;
        }
    }
    else if(nextTurn == PieceColor::black)
    {
        for(const CIS::Square& sq : state[1]->en_passant)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
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
    using CIS = ChessInformationSet;
    
    std::unique_ptr<FullChessInfo> observation = decodeObservation(pos);
    CIS::OnePlayerChessInfo& selfObs = (selfColor==white)?observation->white:observation->black;
    CIS::OnePlayerChessInfo& selfState = playerPiecesTracker;

    bool onePieceCaptured = false;
    std::vector<CIS::BoardClause> conditions;
    
    // test for captured pawns
    auto pawnHere = selfObs.getBlockCheck(selfObs.pawns,CIS::PieceType::pawn);
    auto enPassantHere = selfState.getBlockCheck(selfState.en_passant,CIS::PieceType::pawn);
    for(CIS::Square& sq : selfState.pawns)
    {
        if(!pawnHere(sq))
        // Pawn of self was captured
        {
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
            conditions.push_back(capturedDirect | capturedEnPassant);
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
    StateObj *pos
)
{
    using CIS = ChessInformationSet;
    
    std::unique_ptr<FullChessInfo> observation = decodeObservation(pos);
    if(observation->nextTurn==selfColor || selfColor==PieceColor::empty)
        throw std::logic_error("Wrong turn marker");
    if(observation->currentPhase!=MovePhase::sense)
        throw std::logic_error("Wrong move phase marker");
        
    CIS::OnePlayerChessInfo& selfObs = (selfColor==white)?observation->white:observation->black;
    std::function<std::pair<bool,CIS::PieceType>(const CIS::Square&)> squareToPiece;
    squareToPiece = selfObs.getSquarePieceTypeCheck();
    
    Action selfLastAction = this->evalInfo->bestMove;
    open_spiel::chess::Move selfLastMove = pos->ActionToMove(selfLastAction);
    
    //Test for castling
    bool castling = selfLastMove.is_castling;

    //Test for promotion
    CIS::PieceType promotionType = CIS::OpenSpielPieceType_to_CISPieceType(selfLastMove.promotion_type);
    bool promotion = (promotionType!=CIS::PieceType::empty)?true:false;        

    // Find moved piece and determine the squares
    CIS::Square fromSquare = CIS::Square(selfLastMove.from);
    CIS::Square toSquareAim = CIS::Square(selfLastMove.to);
    std::pair<bool,CIS::PieceType> fromPiece = squareToPiece(fromSquare);
    if(!fromPiece.first)
        throw std::logic_error("Move from empty square");
    CIS::PieceType initialMovePiece = fromPiece.second;

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
        /*
        switch (promotionType)
        {
            case CIS::PieceType::knight:
                currentStateMovePieces = selfObs.knights;
                break;
            case CIS::PieceType::bishop:
                currentStateMovePieces = selfObs.bishops;
                break;
            case CIS::PieceType::rook:
                currentStateMovePieces = selfObs.rooks;
                break;
            case CIS::PieceType::queen:
                currentStateMovePieces = selfObs.queens;
                break;
            default:
                throw std::logic_error("Promotion to invalid piece!");
        }
        */
    }
    else
    {
        std::unordered_set<CIS::Square,CIS::Square::Hasher> previousStateSquareSet(previousStateMovePieces.begin(),previousStateMovePieces.end());
        bool pieceMoved = false;
        for(const CIS::Square& sq : currentStateMovePieces)
        {
            auto iter = previousStateSquareSet.find(sq);
            if(iter!=previousStateSquareSet.end())
            {
                if(pieceMoved)
                    throw std::logic_error("More than one piece of one typeMove from empty square");
                pieceMoved=true;
                toSquareReal = *iter;
            }
        }
        if(!pieceMoved)
            toSquareReal = fromSquare;
    }

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
}

void RBCAgent::handleScanInfo
(
    StateObj *pos,
    ChessInformationSet::Square scanCenter
)
{
    using CIS = ChessInformationSet;
    
    std::unique_ptr<FullChessInfo> observation = decodeObservation(pos);
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
    return randomScanSq;
}

std::unique_ptr<RBCAgent::FullChessInfo> RBCAgent::selectHypothese()
{
    using CIS = ChessInformationSet;
    
    randomHypotheseSelect = std::uniform_int_distribution<std::uint64_t>(0,cis->size());
    std::uint64_t selectedBoard = randomHypotheseSelect(gen);
    auto iter = cis->begin();
    for(int i=0;i<selectedBoard;i++)
    {
        if(iter==cis->end())
            iter = cis->begin();
        else
            iter++;
    }
    std::unique_ptr<std::pair<CIS::OnePlayerChessInfo,double>> selectedHypothese = *iter;
    
    auto fullInfoSet = std::make_unique<FullChessInfo>();
    
    if(selfColor = PieceColor::white)
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
    fullInfoSet->currentPhase = MovePhase::move;
    fullInfoSet->lastMoveCapturedPiece = false;
    fullInfoSet->lastMoveIllegal = false;
    fullInfoSet->nextCompleteTurn = currentTurn;
    
    return fullInfoSet;
}

StateObj* RBCAgent::setupMoveSearchState()
{
    chessOpenSpiel = new OpenSpielState(open_spiel::gametype::SupportedOpenSpielVariants::CHESS);
    std::unique_ptr<FullChessInfo> searchState = selectHypothese();
    chessOpenSpiel->set(searchState->getFEN(),false,open_spiel::gametype::SupportedOpenSpielVariants::CHESS);
    return chessOpenSpiel;
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
    cis->clearRemoved();
    auto newCis = std::make_unique<CIS>();
    for(auto iter=cis->begin(); iter!=cis->end(); iter++)
    {
        CIS::OnePlayerChessInfo& hypoPiecesOpponent = (*iter)->first;
        double probability = (*iter)->second;
        std::unique_ptr<std::vector<std::pair<CIS::OnePlayerChessInfo,double>>> newHypotheses;
        newHypotheses = generateHypotheses(hypoPiecesOpponent);
        for(auto& oneHypothese : *newHypotheses)
            oneHypothese.second = probability;
        cis->remove(iter);
        
        try
        {
            newCis->add(*newHypotheses);
        }
        catch(const std::bad_alloc& e)
        {
            cis->clearRemoved();
            iter = cis->begin();
            newCis->add(*newHypotheses);
        }
    }
    cis = std::move(newCis);
}
