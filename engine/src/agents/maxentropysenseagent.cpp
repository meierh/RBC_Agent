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
 * @file: maxentropysenseagent.cpp
 * Created on 12.2023
 * @author: meierh
 * 
 */

#include <string>
#include <thread>
#include <fstream>
#include "maxentropysenseagent.h"

ChessInformationSet::Square MaxEntropySenseAgent::selectScanAction
(
    StateObj *pos
)
{
    using CIS=ChessInformationSet;
    std::unique_ptr<CIS::Distribution> p_sq_PT = cis->computeDistribution();
    
    std::array<double,64> sensingEntropy;
    std::fill(sensingEntropy.begin(),sensingEntropy.end(),0);
    
    for(auto iter=cis->begin(); iter!=cis->end(); iter++)
    {
        CIS::OnePlayerChessInfo& hypoPiecesOpponent = (*iter)->first;
        std::function<std::pair<bool,CIS::PieceType>(const CIS::Square&)> squareToPiece;
        squareToPiece = hypoPiecesOpponent.getSquarePieceTypeCheck();

        std::array<double,64> p_sq_Hypo;
        for(std::uint8_t ind=0; ind<p_sq_Hypo.size(); ind++)
        {
            std::pair<bool,CIS::PieceType> pieceInfo = squareToPiece(CIS::boardIndexToSquare(ind));
            CIS::PieceType pT = CIS::PieceType::empty;
            if(pieceInfo.first)
                pT = pieceInfo.second;
            p_sq_Hypo[ind] = p_sq_PT->getProbability(ind,pT);
        }

        for(std::uint8_t colI=1; colI<7; colI++)
        {
            for(std::uint8_t rowI=1; rowI<7; rowI++)
            {
                CIS::Square sq(static_cast<CIS::ChessColumn>(colI),static_cast<CIS::ChessRow>(rowI));
                std::pair<std::array<std::uint8_t,9>,std::array<CIS::Square,9>> sensingArea;
                sensingArea = getSensingBoardIndexes(sq);
                double sensingAreaProb=1;
                for(uint i=0; i<sensingArea.first.size(); i++)
                {
                    sensingAreaProb *= p_sq_Hypo[sensingArea.first[i]];
                }
                
                std::uint8_t boardInd = CIS::squareToBoardIndex(sq);
                sensingEntropy[boardInd] += (sensingAreaProb*std::log2(sensingAreaProb));                
            }
        }
    }
    
    std::transform(sensingEntropy.begin(),sensingEntropy.end(),sensingEntropy.begin(),
                   [](double e)->double {return -e;});
    
    std::uint8_t maxEntropyInd=0;
    double maxEntropy = std::numeric_limits<double>::min();
    for(std::uint8_t colI=1; colI<7; colI++)
    {
        for(std::uint8_t rowI=1; rowI<7; rowI++)
        {
            CIS::Square sq(static_cast<CIS::ChessColumn>(colI),static_cast<CIS::ChessRow>(rowI));                
            std::uint8_t boardInd = CIS::squareToBoardIndex(sq);
            
            if(maxEntropy<sensingEntropy[boardInd])
            {
                maxEntropy = sensingEntropy[boardInd];
                maxEntropyInd = boardInd;
            }
        }
    }
    
    CIS::Square maxEntropySq = CIS::boardIndexToSquare(maxEntropyInd);
    return maxEntropySq;
}
