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
    auto hypotheses = std::make_unique<std::vector<ChessInformationSet::ChessPiecesInformation>>();
    
    std::function<bool(const CIS_Square&)> piecesSelfBlock = piecesSelf.getBlockCheck();
    std::function<bool(const CIS_Square&)> piecesOppoBlock = piecesOpponent.getBlockCheck();
    
    auto pawnLegalMoves = [&](const CIS_Square& sq)
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
    auto rookLegalMoves = [&](const CIS_Square& sq)
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
    auto knightLegalMoves = [&](const CIS_Square& sq)
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
    auto bishopLegalMoves = [&](const CIS_Square& sq)
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
    auto queenLegalMoves = [&](const CIS_Square& sq)
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
    auto kingLegalMoves = [&](const CIS_Square& sq)
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
    auto obs = std::make_unique<ChessPiecesObservation>();
    
    // decode here
    
    return obs;
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



/*
string RBCAgent::get_name() const
{
    return MCTSAgentBatch::get_name();
}

void RBCAgent::evaluate_board_state()
{
    MCTSAgentBatch::evaluate_board_state();
}
*/
