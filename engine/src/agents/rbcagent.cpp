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

/*
#include "../evalinfo.h"
#include "../constants.h"
#include "../util/blazeutil.h"
#include "../manager/treemanager.h"
#include "../manager/threadmanager.h"
#include "../node.h"
#include "../util/communication.h"
#include "util/gcthread.h"
*/


RBCAgent::RBCAgent
(
    NeuralNetAPI *netSingle,
    vector<unique_ptr<NeuralNetAPI>>& netBatches,
    SearchSettings* searchSettings,
    PlaySettings* playSettings,
    int noa,
    bool sN
):
MCTSAgentBatch(netSingle, netBatches, searchSettings, playSettings, noa, sN),
currentTurn(0)
{}

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
    handleOpponentMoveInfo(pos);
    ChessInformationSet::Square scanCenter = applyScanAction(pos);
    handleScanInfo(pos,scanCenter);
    
    MCTSAgentBatch::set_search_settings(pos,searchLimits,evalInfo);
}

std::unique_ptr<std::vector<ChessInformationSet::OnePlayerChessInfo>> RBCAgent::generateHypotheses
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
    auto hypotheses = std::make_unique<std::vector<CIS_CPI>>();
    
    FullChessInfo fullState;
    PieceColor opponentColor;
    unsigned int nextCompleteTurn;
    if(selfColor == PieceColor::white)
    {
        fullState.white = piecesSelf;
        fullState.black = piecesOpponent;
        opponentColor = PieceColor::black;
        nextCompleteTurn = currentTurn;
    }
    else
    {
        fullState.white = piecesOpponent;
        fullState.black = piecesSelf;
        opponentColor = PieceColor::white;
        nextCompleteTurn = currentTurn+1;
    }
    std::string fen = fullState.getFEN(opponentColor,nextCompleteTurn);
    OpenSpielState hypotheticState;
    hypotheticState.set(fen,false,StateConstants::DEFAULT_VARIANT());
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
        
        hypotheses->push_back(piecesOpponent);
        CIS::OnePlayerChessInfo& new_hypothese = hypotheses->back();
        
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

std::unique_ptr<std::vector<ChessInformationSet::OnePlayerChessInfo>> RBCAgent::generateHypotheses
(
    ChessInformationSet::OnePlayerChessInfo& piecesOpponent
)
{
    return generateHypotheses(piecesOpponent,this->playerPiecesTracker,this->selfColor);
}

std::unique_ptr<RBCAgent::FullChessInfo> RBCAgent::decodeStatePlane
(
    StateObj *pos
) const
{
    using CIS = ChessInformationSet;
    
    auto inputPlanesSmPtr = std::make_unique<float[]>(net->get_batch_size() * net->get_nb_input_values_total());   
    float* inputPlanes =  inputPlanesSmPtr.get();
    pos->get_state_planes(true,inputPlanes,1);
    
    std::uint16_t offset = 0;    
    auto info = std::make_unique<FullChessInfo>();
    std::array<CIS::OnePlayerChessInfo*,2> obs = {&(info->white),&(info->black)};
    
    //pieces position
    for(std::uint16_t color=0; color<obs.size(); color++)
    {
        std::array<float,64> pawns;
        std::memcpy(pawns.data(),pos+offset,64);
        for(unsigned int index=0; index<pawns.size(); index++)
        {
            if(pawns[index]>0.5)
            {
                obs[color]->pawns.push_back(CIS::boardIndexToSquare(index));
            }
        }
        if(obs[color]->pawns.size()>8)
            std::logic_error("Can not have more than 8 pawns");
        offset+=64;

        
        std::array<float,64> knights;
        std::memcpy(knights.data(),pos+offset,64);
        for(unsigned int index=0; index<knights.size(); index++)
        {
            if(knights[index]>0.5)
            {
                obs[color]->knights.push_back(CIS::boardIndexToSquare(index));
            }
        }
        if(obs[color]->knights.size()>2)
            std::logic_error("Can not have more than 2 knights");
        offset+=64;
        
        std::array<float,64> bishops;
        std::memcpy(bishops.data(),pos+offset,64);
        for(unsigned int index=0; index<bishops.size(); index++)
        {
            if(bishops[index]>0.5)
            {
                obs[color]->bishops.push_back(CIS::boardIndexToSquare(index));
            }
        }
        if(obs[color]->bishops.size()>2)
            std::logic_error("Can not have more than 2 bishops");
        offset+=64;
        
        std::array<float,64> rooks;
        std::memcpy(rooks.data(),pos+offset,64);
        for(unsigned int index=0; index<rooks.size(); index++)
        {
            if(rooks[index]>0.5)
            {
                obs[color]->rooks.push_back(CIS::boardIndexToSquare(index));
            }
        }
        if(obs[color]->rooks.size()>2)
            std::logic_error("Can not have more than 2 rooks");
        offset+=64;
        
        std::array<float,64> queens;
        std::memcpy(queens.data(),pos+offset,64);
        for(unsigned int index=0; index<queens.size(); index++)
        {
            if(queens[index]>0.5)
            {
                obs[color]->queens.push_back(CIS::boardIndexToSquare(index));
            }
        }
        if(obs[color]->queens.size()>1)
            std::logic_error("Can not have more than 1 queens");
        offset+=64;
        
        std::array<float,64> kings;
        std::memcpy(kings.data(),pos+offset,64);
        for(unsigned int index=0; index<kings.size(); index++)
        {
            if(kings[index]>0.5)
            {
                obs[color]->kings.push_back(CIS::boardIndexToSquare(index));
            }
        }
        if(obs[color]->kings.size()>1)
            std::logic_error("Can not have more than 1 kings");
        offset+=64;
    }
    
    offset += 2*64;  // repetitions 1&2
    offset += 5*64;  // white pocket pieces
    offset += 5*64;  // black pocket pieces
    offset += 2*64;  // white&black promotions
    
    std::array<float,64> en_passant;
    std::memcpy(en_passant.data(),pos+offset,64);
    // Decode en_passent here
    
    offset += 1*64;  // turn color
    offset += 1*64;  // total move communication
    
    for(std::uint16_t color=0; color<obs.size(); color++)
    {        
        std::array<float,64> castle_king_side;
        std::memcpy(castle_king_side.data(),pos+offset,64);
        if(castle_king_side[0]>0.5)
        {
            obs[color]->kingside=true;
        }
        else
        {
            obs[color]->kingside=false;
        }
        offset+=64;
        
        std::array<float,64> castle_queen_side;
        std::memcpy(castle_queen_side.data(),pos+offset,64);
        if(castle_queen_side[0]>0.5)
        {
            obs[color]->queenside=true;
        }
        else
        {
            obs[color]->queenside=false;
        }
        offset+=64;
    }
    
    std::array<float,64> no_progress_count;
    std::memcpy(no_progress_count.data(),pos+offset,64);
    obs[0]->no_progress_count=static_cast<std::uint8_t>(no_progress_count[0]);
    obs[1]->no_progress_count=static_cast<std::uint8_t>(no_progress_count[0]);
    
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
    
    std::unique_ptr<FullChessInfo> observation = decodeStatePlane(pos);
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
    
    cis.markIncompatibleBoards(conditions);
}

void RBCAgent::handleScanInfo
(
    StateObj *pos,
    ChessInformationSet::Square scanCenter
)
{
    using CIS = ChessInformationSet;
    
    std::unique_ptr<FullChessInfo> observation = decodeStatePlane(pos);
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
    cis.markIncompatibleBoards(conditions);
}

ChessInformationSet::Square RBCAgent::applyScanAction
(
    StateObj *pos
)
{
    return {ChessInformationSet::ChessColumn::A,ChessInformationSet::ChessRow::one}; // dummy
}
