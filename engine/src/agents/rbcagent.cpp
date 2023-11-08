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
MCTSAgentBatch(netSingle, netBatches, searchSettings, playSettings, noa, sN)
{}

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

std::unique_ptr<std::vector<ChessInformationSet::ChessPiecesInformation>> RBCAgent::generateHypotheses
(
    ChessInformationSet::ChessPiecesInformation& piecesOpponent,
    ChessInformationSet::ChessPiecesInformation& piecesSelf,
    const RBCAgent::PieceColor selfColor
) const
{
    using CIS_Square = ChessInformationSet::Square;
    using CIS_CPI = ChessInformationSet::ChessPiecesInformation;
    auto hypotheses = std::make_unique<std::vector<CIS_CPI>>();
    
    std::function<bool(const CIS_Square&)> piecesSelfBlock = piecesSelf.getBlockCheck();
    std::function<bool(const CIS_Square&)> piecesOppoBlock = piecesOpponent.getBlockCheck();
    
    std::function<std::unique_ptr<std::vector<CIS_Square>>(const CIS_Square&)>
    pawnLegalMoves = [&](const CIS_Square& sq)
    {
        bool hasNotMoved=false;
        if(selfColor == PieceColor::White)
        {
            if(sq.row == ChessInformationSet::ChessRow::one)
                std::logic_error("White pawn can not be on row one");
            else if(sq.row == ChessInformationSet::ChessRow::two)
                hasNotMoved=true;
        }
        else
        {
            if(sq.row == ChessInformationSet::ChessRow::eight)
                std::logic_error("Black pawn can not be on row eight");
            else if(sq.row == ChessInformationSet::ChessRow::seven)
                hasNotMoved=true;
        }
        
        auto legalDestinations = std::make_unique<std::vector<CIS_Square>>();
        if(selfColor == PieceColor::White)
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
    
    return hypotheses;
}

std::unique_ptr<std::vector<ChessInformationSet::ChessPiecesInformation>> RBCAgent::generateHypotheses
(
    ChessInformationSet::ChessPiecesInformation& piecesOpponent
)
{
    return generateHypotheses(piecesOpponent,this->playerPiecesTracker,this->selfColor);
}

std::unique_ptr<RBCAgent::ChessPiecesObservation> RBCAgent::getDecodedStatePlane
(
    StateObj *pos,
    const Player side  
) const
{
    float* inputPlanes;
    pos->get_state_planes(true,inputPlanes,1);
    
    std::function<ChessInformationSet::Square(std::uint8_t index)> indexToSquare;
    indexToSquare = [](std::uint8_t index)
    {
        std::uint8_t x = index / 8;
        std::uint8_t y = index % 8;
        ChessInformationSet::Square sq;
        sq.column = static_cast<ChessInformationSet::ChessColumn>(x);
        sq.row = static_cast<ChessInformationSet::ChessRow>(y);
        return sq;
    };
    
    std::uint16_t offset = 0;
    
    std::array<std::unique_ptr<ChessPiecesObservation>,2> obs;
    obs[0] = std::make_unique<ChessPiecesObservation>(); //white
    obs[1] = std::make_unique<ChessPiecesObservation>(); //black
    
    for(std::uint16_t color=0; color<obs.size(); color++)
    {
        std::array<float,64> pawns;
        std::memcpy(pawns.data(),pos+offset,64);
        for(unsigned int index=0; index<pawns.size(); index++)
        {
            if(pawns[index]>0.5)
            {
                obs[color]->pawns.push_back(indexToSquare(index));
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
                obs[color]->knights.push_back(indexToSquare(index));
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
                obs[color]->bishops.push_back(indexToSquare(index));
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
                obs[color]->rooks.push_back(indexToSquare(index));
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
                obs[color]->queens.push_back(indexToSquare(index));
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
                obs[color]->kings.push_back(indexToSquare(index));
            }
        }
        if(obs[color]->kings.size()>1)
            std::logic_error("Can not have more than 1 kings");
        offset+=64;
    }
    
    //obsWhite;
}

void RBCAgent::handleOpponentMoveInfo
(
    StateObj *pos
)
{
    std::vector<ChessInformationSet::Square> captureSquares;
    std::unique_ptr<ChessPiecesObservation> ownPiecesObs = getDecodedStatePlane(pos,Player::Self);
    std::unordered_map<std::uint8_t,std::vector<ChessInformationSet::Square>*> comparisonSet = 
    {
        {0,&(ownPiecesObs->pawns)},{1,&(ownPiecesObs->pawns)},{2,&(ownPiecesObs->pawns)},{3,&(ownPiecesObs->pawns)},{4,&(ownPiecesObs->pawns)},{5,&(ownPiecesObs->pawns)},{6,&(ownPiecesObs->pawns)},{7,&(ownPiecesObs->pawns)},
        {8,&(ownPiecesObs->rooks)},{15,&(ownPiecesObs->rooks)},
        {9,&(ownPiecesObs->knights)},{14,&(ownPiecesObs->knights)},
        {10,&(ownPiecesObs->bishops)},{13,&(ownPiecesObs->bishops)},
        {11,&(ownPiecesObs->queens)},
        {12,&(ownPiecesObs->kings)}
    };
    
    // Test for captured pawns
    for(unsigned int i=0; i<16; i++) 
    {
        std::pair<ChessInformationSet::Square,bool>& onePiece = playerPiecesTracker.data[i];
        if(onePiece.second)
        {
            bool pieceExist = false;
            std::vector<ChessInformationSet::Square>* set = comparisonSet[i];
            for(ChessInformationSet::Square& obsOwnPawn : *set)
            {
                if(obsOwnPawn == onePiece.first)
                    pieceExist = true;
            }
            if(!pieceExist)
            {
                captureSquares.push_back(onePiece.first);
                onePiece.second = false;
            }
        }
    }
    
    if(captureSquares.size()>1)
    {
        // Throw error
    }
    else if(captureSquares.size()==1)
    {
        std::vector<ChessInformationSet::Square> noPieces;
        std::vector<std::pair<ChessInformationSet::PieceType,ChessInformationSet::Square>> knownPieces;
        cis.markIncompatibleBoards(noPieces,captureSquares,knownPieces);
    }
}

void RBCAgent::handleScanInfo
(
    StateObj *pos,
    ChessInformationSet::Square scanCenter
)
{
    std::unique_ptr<ChessPiecesObservation> opponentPiecesObs = getDecodedStatePlane(pos,Player::Opponent);
    
    std::unordered_set<ChessInformationSet::Square,ChessInformationSet::Square::Hasher> scannedSquares;
    int colC = static_cast<int>(scanCenter.column);
    int rowC = static_cast<int>(scanCenter.row);
    for(int col=colC-1; col<colC+2; col++)
    {
        for(int row=rowC-1; col<rowC+2; row++)
        {
            if(row>=0 && row<8)
            {
                if(col>=0 && row<8)
                {
                    ChessInformationSet::ChessColumn c = static_cast<ChessInformationSet::ChessColumn>(col);
                    ChessInformationSet::ChessRow r = static_cast<ChessInformationSet::ChessRow>(row);
                    scannedSquares.insert({c,r});
                }
            }
        }
    }

    std::vector<ChessInformationSet::Square> noPieces;
    std::vector<ChessInformationSet::Square> unknownPieces;
    std::vector<std::pair<ChessInformationSet::PieceType,ChessInformationSet::Square>> knownPieces;
    
    for(ChessInformationSet::Square sq : opponentPiecesObs->pawns)
        knownPieces.push_back({ChessInformationSet::PieceType::pawn,sq});
    
    for(ChessInformationSet::Square sq : opponentPiecesObs->knights)
        knownPieces.push_back({ChessInformationSet::PieceType::knight,sq});
    
    for(ChessInformationSet::Square sq : opponentPiecesObs->bishops)
        knownPieces.push_back({ChessInformationSet::PieceType::bishop,sq});
    
    for(ChessInformationSet::Square sq : opponentPiecesObs->rooks)
        knownPieces.push_back({ChessInformationSet::PieceType::rook,sq});
    
    for(ChessInformationSet::Square sq : opponentPiecesObs->queens)
        knownPieces.push_back({ChessInformationSet::PieceType::queen,sq});
    
    for(ChessInformationSet::Square sq : opponentPiecesObs->kings)
        knownPieces.push_back({ChessInformationSet::PieceType::king,sq});
    
    for(std::pair<ChessInformationSet::PieceType,ChessInformationSet::Square> knownPiece : knownPieces)
    {
        auto iter = scannedSquares.find(knownPiece.second);
        if(iter!=scannedSquares.end())
        {
            scannedSquares.erase(iter);
        }
    }
    
    for(auto iter = scannedSquares.begin(); iter!=scannedSquares.end(); iter++)
    {
        noPieces.push_back(*iter);
    }
    
    cis.markIncompatibleBoards(noPieces,unknownPieces,knownPieces);
}

ChessInformationSet::Square RBCAgent::applyScanAction
(
    StateObj *pos
)
{
    return {ChessInformationSet::ChessColumn::A,ChessInformationSet::ChessRow::one}; // dummy
}
