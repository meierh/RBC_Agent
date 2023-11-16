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
}

RBCAgent::PieceColor RBCAgent::OpenSpielColor_to_RBCColor
(
    const open_spiel::chess::Color os_pC
)
{
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
                throw std::logic_error("Pieces overlay!");
            unsigned int column = static_cast<unsigned int>(sq.column);
            unsigned int row = static_cast<unsigned int>(sq.row);
            char columnChar = column + 97;
            char rowChar = row + 60;
            enPassantString += columnChar;
            enPassantString += rowChar;
        }
        
        charOffset+=32;
    }
    
    if(white.no_progress_count != black.no_progress_count)
        throw std::logic_error("Mismatch in no progress count!");
    halfTurns = white.no_progress_count;
        
    //Build string
    std::string piecePlacement;
    std::vector<std::string> piecePlacementRows;
    for(std::array<char,8> oneRow : chessBoard)
    {
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
        CIS::PieceType promPieceType = CIS::OpenSpielPieceType_to_CISPieceType(move.promotion_type);
    }
    
    //open_spiel::chess::Move ActionToMove(Action action)
    
    
    
    
    /*
    std::function<bool(const CIS_Square&)> piecesSelfBlock = piecesSelf.getBlockCheck();
    std::function<bool(const CIS_Square&)> piecesOppoBlock = piecesOpponent.getBlockCheck();
    
    std::function<std::unique_ptr<std::vector<CIS_Square>>(const CIS_Square&)>
    pawnLegalMoves = [&](const CIS_Square& sq)
    {
        bool hasNotMoved=false;
        if(selfColor == PieceColor::white)
        {
            if(sq.row == ChessInformationSet::ChessRow::one)
                std::logic_error("white pawn can not be on row one");
            else if(sq.row == ChessInformationSet::ChessRow::two)
                hasNotMoved=true;
        }
        else
        {
            if(sq.row == ChessInformationSet::ChessRow::eight)
                std::logic_error("black pawn can not be on row eight");
            else if(sq.row == ChessInformationSet::ChessRow::seven)
                hasNotMoved=true;
        }
        
        auto legalDestinations = std::make_unique<std::vector<CIS_Square>>();
        if(selfColor == PieceColor::white)
        {
            // Non capturing moves
            CIS_Square oneForw(sq);
            if(oneForw.vertPlus(1) && !piecesSelfBlock(oneForw) && !piecesOppoBlock(oneForw))
            {
                legalDestinations->push_back(oneForw);
                if(hasNotMoved && oneForw.vertPlus(1) && !piecesSelfBlock(oneForw) && !piecesOppoBlock(oneForw))
                    legalDestinations->push_back(oneForw);
            }
            
            // Capturing moves
            CIS_Square captu(sq);
            if(captu.diagVertPlusHorizMinus(1) && piecesSelfBlock(captu))
                legalDestinations->push_back(captu);
            captu = sq;
            if(captu.diagVertPlusHorizPlus(1) && piecesSelfBlock(captu))
                legalDestinations->push_back(captu);
        }
        else
        {
            // Non capturing moves
            CIS_Square oneForw(sq);
            if(oneForw.vertMinus(1) && !piecesSelfBlock(oneForw) && !piecesOppoBlock(oneForw))
            {
                legalDestinations->push_back(oneForw);
                if(hasNotMoved && oneForw.vertMinus(1) && !piecesSelfBlock(oneForw) && !piecesOppoBlock(oneForw))
                    legalDestinations->push_back(oneForw);
            }
            
            // Capturing moves
            CIS_Square captu(sq);
            if(captu.diagVertMinusHorizMinus(1) && piecesSelfBlock(captu))
                legalDestinations->push_back(captu);
            captu = sq;
            if(captu.diagVertMinusHorizPlus(1) && piecesSelfBlock(captu))
                legalDestinations->push_back(captu);
        }
        return legalDestinations;
    };
    
    std::function<std::unique_ptr<std::vector<CIS_Square>>(const CIS_Square&)>
    rookLegalMoves = [&](const CIS_Square& sq)
    {       
        auto legalDestinations = std::make_unique<std::vector<CIS_Square>>();
        
        std::array<CIS_Square,4> moveFront = {sq,sq,sq,sq};        
        std::bitset<4> moveFrontBlocked(0000);
        while(!moveFrontBlocked.all())
        {
            if(!moveFrontBlocked[0] && moveFront[0].vertPlus(1) && !piecesOppoBlock(moveFront[0]))
            {
                legalDestinations->push_back(moveFront[0]);
                if(piecesSelfBlock(moveFront[0]))
                    moveFrontBlocked[0]=true;
            }
            else
                moveFrontBlocked[0]=true;

            if(!moveFrontBlocked[1] && moveFront[1].vertMinus(1) && !piecesOppoBlock(moveFront[1]))
            {
                legalDestinations->push_back(moveFront[1]);
                if(piecesSelfBlock(moveFront[1]))
                    moveFrontBlocked[1]=true;
            }
            else
                moveFrontBlocked[1]=true;
          
            if(!moveFrontBlocked[2] && moveFront[2].horizPlus(1) && !piecesOppoBlock(moveFront[2]))
            {
                legalDestinations->push_back(moveFront[2]);
                if(piecesSelfBlock(moveFront[2]))
                    moveFrontBlocked[2]=true;
            }
            else
                moveFrontBlocked[2]=true;
                
            if(!moveFrontBlocked[3] && moveFront[3].horizMinus(1) && !piecesOppoBlock(moveFront[3]))
            {
                legalDestinations->push_back(moveFront[3]);
                if(piecesSelfBlock(moveFront[3]))
                    moveFrontBlocked[3]=true;
            }
            else
                moveFrontBlocked[3]=true;
        }
        return legalDestinations;
    };

    std::function<std::unique_ptr<std::vector<CIS_Square>>(const CIS_Square&)>
    knightLegalMoves = [&](const CIS_Square& sq)
    {       
        auto legalDestinations = std::make_unique<std::vector<CIS_Square>>();
        
        std::array<CIS_Square,8> moveFront = {sq,sq,sq,sq,sq,sq,sq,sq};        

        if(moveFront[0].knightVertPlusHorizPlus() && !piecesOppoBlock(moveFront[0]))
            legalDestinations->push_back(moveFront[0]);

        if(moveFront[1].knightVertPlusHorizMinus() && !piecesOppoBlock(moveFront[1]))
            legalDestinations->push_back(moveFront[1]);
        
        if(moveFront[2].knightVertMinusHorizPlus() && !piecesOppoBlock(moveFront[2]))
            legalDestinations->push_back(moveFront[2]);
            
        if(moveFront[3].knightVertMinusHorizMinus() && !piecesOppoBlock(moveFront[3]))
            legalDestinations->push_back(moveFront[3]);
        
        if(moveFront[4].knightHorizPlusVertPlus() && !piecesOppoBlock(moveFront[4]))
            legalDestinations->push_back(moveFront[4]);

        if(moveFront[5].knightHorizPlusVertMinus() && !piecesOppoBlock(moveFront[5]))
            legalDestinations->push_back(moveFront[5]);
        
        if(moveFront[6].knightHorizMinusVertPlus() && !piecesOppoBlock(moveFront[6]))
            legalDestinations->push_back(moveFront[6]);
            
        if(moveFront[7].knightHorizMinusVertMinus() && !piecesOppoBlock(moveFront[7]))
            legalDestinations->push_back(moveFront[7]);
        
        return legalDestinations;        
    };
    
    std::function<std::unique_ptr<std::vector<CIS_Square>>(const CIS_Square&)>
    bishopLegalMoves = [&](const CIS_Square& sq)
    {       
        auto legalDestinations = std::make_unique<std::vector<CIS_Square>>();
        
        std::array<CIS_Square,4> moveFront = {sq,sq,sq,sq};        
        std::bitset<4> moveFrontBlocked(0000);
        while(!moveFrontBlocked.all())
        {
            if(!moveFrontBlocked[0] && moveFront[0].diagVertPlusHorizPlus(1) && !piecesOppoBlock(moveFront[0]))
            {
                legalDestinations->push_back(moveFront[0]);
                if(piecesSelfBlock(moveFront[0]))
                    moveFrontBlocked[0]=true;
            }
            else
                moveFrontBlocked[0]=true;

            if(!moveFrontBlocked[1] && moveFront[1].diagVertMinusHorizPlus(1) && !piecesOppoBlock(moveFront[1]))
            {
                legalDestinations->push_back(moveFront[1]);
                if(piecesSelfBlock(moveFront[1]))
                    moveFrontBlocked[1]=true;
            }
            else
                moveFrontBlocked[1]=true;
          
            if(!moveFrontBlocked[2] && moveFront[2].diagVertPlusHorizMinus(1) && !piecesOppoBlock(moveFront[2]))
            {
                legalDestinations->push_back(moveFront[2]);
                if(piecesSelfBlock(moveFront[2]))
                    moveFrontBlocked[2]=true;
            }
            else
                moveFrontBlocked[2]=true;
                
            if(!moveFrontBlocked[3] && moveFront[3].diagVertMinusHorizMinus(1) && !piecesOppoBlock(moveFront[3]))
            {
                legalDestinations->push_back(moveFront[3]);
                if(piecesSelfBlock(moveFront[3]))
                    moveFrontBlocked[3]=true;
            }
            else
                moveFrontBlocked[3]=true;
        }
        return legalDestinations;
    };
    
    std::function<std::unique_ptr<std::vector<CIS_Square>>(const CIS_Square&)>
    queenLegalMoves = [&](const CIS_Square& sq)
    {       
        auto legalDestinations = std::make_unique<std::vector<CIS_Square>>();
        
        std::array<CIS_Square,8> moveFront = {sq,sq,sq,sq,sq,sq,sq,sq};        
        std::bitset<8> moveFrontBlocked(00000000);
        while(!moveFrontBlocked.all())
        {
            if(!moveFrontBlocked[0] && moveFront[0].vertPlus(1) && !piecesOppoBlock(moveFront[0]))
            {
                legalDestinations->push_back(moveFront[0]);
                if(piecesSelfBlock(moveFront[0]))
                    moveFrontBlocked[0]=true;
            }
            else
                moveFrontBlocked[0]=true;

            if(!moveFrontBlocked[1] && moveFront[1].vertMinus(1) && !piecesOppoBlock(moveFront[1]))
            {
                legalDestinations->push_back(moveFront[1]);
                if(piecesSelfBlock(moveFront[1]))
                    moveFrontBlocked[1]=true;
            }
            else
                moveFrontBlocked[1]=true;
          
            if(!moveFrontBlocked[2] && moveFront[2].horizPlus(1) && !piecesOppoBlock(moveFront[2]))
            {
                legalDestinations->push_back(moveFront[2]);
                if(piecesSelfBlock(moveFront[2]))
                    moveFrontBlocked[2]=true;
            }
            else
                moveFrontBlocked[2]=true;
                
            if(!moveFrontBlocked[3] && moveFront[3].horizMinus(1) && !piecesOppoBlock(moveFront[3]))
            {
                legalDestinations->push_back(moveFront[3]);
                if(piecesSelfBlock(moveFront[3]))
                    moveFrontBlocked[3]=true;
            }
            else
                moveFrontBlocked[3]=true;
            
            if(!moveFrontBlocked[4] && moveFront[4].diagVertPlusHorizPlus(1) && !piecesOppoBlock(moveFront[4]))
            {
                legalDestinations->push_back(moveFront[4]);
                if(piecesSelfBlock(moveFront[4]))
                    moveFrontBlocked[4]=true;
            }
            else
                moveFrontBlocked[4]=true;

            if(!moveFrontBlocked[5] && moveFront[5].diagVertMinusHorizPlus(1) && !piecesOppoBlock(moveFront[5]))
            {
                legalDestinations->push_back(moveFront[5]);
                if(piecesSelfBlock(moveFront[5]))
                    moveFrontBlocked[5]=true;
            }
            else
                moveFrontBlocked[5]=true;
          
            if(!moveFrontBlocked[6] && moveFront[6].diagVertPlusHorizMinus(1) && !piecesOppoBlock(moveFront[6]))
            {
                legalDestinations->push_back(moveFront[6]);
                if(piecesSelfBlock(moveFront[6]))
                    moveFrontBlocked[6]=true;
            }
            else
                moveFrontBlocked[6]=true;
                
            if(!moveFrontBlocked[7] && moveFront[7].diagVertMinusHorizMinus(1) && !piecesOppoBlock(moveFront[7]))
            {
                legalDestinations->push_back(moveFront[7]);
                if(piecesSelfBlock(moveFront[7]))
                    moveFrontBlocked[7]=true;
            }
            else
                moveFrontBlocked[7]=true;
        }
        return legalDestinations;
    };
    
    std::function<std::unique_ptr<std::vector<CIS_Square>>(const CIS_Square&)>
    kingLegalMoves = [&](const CIS_Square& sq)
    {       
        auto legalDestinations = std::make_unique<std::vector<CIS_Square>>();
        
        std::array<CIS_Square,8> moveFront = {sq,sq,sq,sq,sq,sq,sq,sq};        

        if(moveFront[0].vertPlus(1) && !piecesOppoBlock(moveFront[0]))
            legalDestinations->push_back(moveFront[0]);

        if(moveFront[1].vertMinus(1) && !piecesOppoBlock(moveFront[1]))
            legalDestinations->push_back(moveFront[1]);
        
        if(moveFront[2].horizPlus(1) && !piecesOppoBlock(moveFront[2]))
            legalDestinations->push_back(moveFront[2]);
            
        if(moveFront[3].horizMinus(1) && !piecesOppoBlock(moveFront[3]))
            legalDestinations->push_back(moveFront[3]);
        
        if(moveFront[4].diagVertPlusHorizPlus(1) && !piecesOppoBlock(moveFront[4]))
            legalDestinations->push_back(moveFront[4]);

        if(moveFront[5].diagVertMinusHorizPlus(1) && !piecesOppoBlock(moveFront[5]))
            legalDestinations->push_back(moveFront[5]);
        
        if(moveFront[6].diagVertPlusHorizMinus(1) && !piecesOppoBlock(moveFront[6]))
            legalDestinations->push_back(moveFront[6]);
            
        if(moveFront[7].diagVertMinusHorizMinus(1) && !piecesOppoBlock(moveFront[7]))
            legalDestinations->push_back(moveFront[7]);
        
        return legalDestinations;
    };
    
    //hypotheses for every figure movement
    for(std::uint8_t pieceInd=0; pieceInd<piecesOpponent.data.size(); pieceInd++)
    {
        const std::pair<CIS_Square,bool>& onePiece = piecesOpponent.data[pieceInd];
        if(onePiece.second)
        {
            std::unique_ptr<std::vector<CIS_Square>> posMoves;
            if(pieceInd<8)
                posMoves = pawnLegalMoves(onePiece.first);
            else if(pieceInd==8 || pieceInd==15)
                posMoves = rookLegalMoves(onePiece.first);
            else if(pieceInd==9 || pieceInd==14)
                posMoves = knightLegalMoves(onePiece.first);
            else if(pieceInd==10 || pieceInd==13)
                posMoves = bishopLegalMoves(onePiece.first);
            else if(pieceInd==11)
                posMoves = queenLegalMoves(onePiece.first);
            else if(pieceInd==12)
                posMoves = kingLegalMoves(onePiece.first);
        
            std::vector<CIS_CPI> onePieceHypotheses(posMoves->size());
            for(unsigned int moveInd=0;moveInd<posMoves->size();moveInd++)
            {
                onePieceHypotheses[moveInd] = piecesOpponent;
                onePieceHypotheses[moveInd].data[pieceInd].first = (*posMoves)[moveInd];
            }
            hypotheses->insert(hypotheses->end(),onePieceHypotheses.begin(),onePieceHypotheses.end());
        }
    } 
    */
    
    return hypotheses;
}

std::unique_ptr<std::vector<ChessInformationSet::OnePlayerChessInfo>> RBCAgent::generateHypotheses
(
    ChessInformationSet::OnePlayerChessInfo& piecesOpponent
)
{
    return generateHypotheses(piecesOpponent,this->playerPiecesTracker,this->selfColor);
}

std::unique_ptr<RBCAgent::FullChessInfo> RBCAgent::getDecodedStatePlane
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

void RBCAgent::handleOpponentMoveInfo
(
    StateObj *pos
)
{
    using CIS = ChessInformationSet;
    
    std::unique_ptr<FullChessInfo> observation = getDecodedStatePlane(pos);
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
    
    std::unique_ptr<FullChessInfo> observation = getDecodedStatePlane(pos);
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
