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
 * @file: informationset.h
 * Created on 17.10.2023
 * @author: meierh
 */

#ifndef CHESSINFORMATIONSET_H
#define CHESSINFORMATIONSET_H

#include <cassert>
#include <memory>
#include <cstdint>
#include <queue>
#include <limits>
#include "informationset.h"

namespace crazyara {
    
constexpr std::uint64_t chessInfoSize=128;

/**
 * @brief The ChessInformationSet class defines a information set to store a chess game state.
 */
class ChessInformationSet : public InformationSet<chessInfoSize>
{
    public:
        enum class ChessColumn {A=0,B=1,C=2,D=3,E=4,F=5,G=6,H=7};
        enum class ChessRow {one=0,two=1,three=2,four=3,five=4,six=5,seven=6,eight=7};
        class Square
        {
        public:
            ChessColumn column;
            ChessRow row;
            bool operator==(const Square& other) const
            {
                return column==other.column && row==other.row;
            }
        };
        enum class Piece {pawn1=0,pawn2=1,pawn3=2,pawn4=3,pawn5=4,pawn6=5,pawn7=6,pawn8=7,
                          rook1=8,knight1=9,bishop1=10,queen=11,king=12,bishop2=13,knight2=14,rook2=15};
        enum class PieceType {pawn=0,rook=1,knight=2,bishop=3,queen=4,king=5};
        
        PieceType boardIndexToPieceType(std::uint8_t boardIndex);
        
        /**
         * Marks boards incompatible with observations
         * @param noPieces
         * @param unknownPieces
         * @param knownPieces
         */
        void markIncompatibleBoards(std::vector<Square>& noPieces, std::vector<Square>& unknownPieces, std::vector<std::pair<PieceType,Square>>& knownPieces);
        
        ChessInformationSet();        
    protected:
        /**
         * Sets the pieces in a given board
         * @param pieces: array of 16 pieces in the following order 
         *          {8*pawn left to right in initial placement,
         *          rook,knight,bishop,queen,king,bishop,knight,rook}
         *          The first item represents the square on the board
         *          The second item represents wether the pieces is not captured
         *          (captured:false, not captured:true)
         * @param probability: probability of given board [0,1]
         * @param index: index of the given board in the infoSet
         */
        void setBoard(const std::array<std::pair<Square,bool>,16> pieces, const double probability, const std::uint64_t index);
        
        /**
         * Gets the pieces in a given board
         * @param index: index of the given board in the infoSet
         * @return pieces: array of 16 pieces in the following order 
         *              {8*pawn left to right in initial placement,
         *              rook,knight,bishop,queen,king,bishop,knight,rook}
         *              The first item represents the square on the board
         *              The second item represents wether the pieces is not captured
         *              (captured:false, not captured:true)
         *         probability of given board [0,1]
         */
        std::unique_ptr<std::pair<std::array<std::pair<Square,bool>,16>,double>> getBoard(const std::uint64_t index) const;
        
        std::unique_ptr<std::pair<std::array<std::pair<Square,bool>,16>,double>> decodeBoard
        (
            const std::bitset<chessInfoSize>& bits
        ) const;
        
        std::unique_ptr<std::bitset<chessInfoSize>> encodeBoard
        (
            const std::array<std::pair<Square,bool>,16>& pieces,
            const double probability
        ) const;

        
        class CIS_Iterator : public IS_Iterator
        {
            public:
                CIS_Iterator
                (
                    ChessInformationSet* cis,
                    std::uint64_t itemInd
                ):
                IS_Iterator(cis,itemInd),
                cis(cis){};
                
                CIS_Iterator
                (
                    ChessInformationSet* cis
                ):
                IS_Iterator(cis),
                cis(cis){};               

                std::unique_ptr<std::pair<std::array<std::pair<Square,bool>,16>,double>> operator*() const noexcept
                {
                    std::unique_ptr<std::bitset<chessInfoSize>> bits = IS_Iterator::operator*();
                    return cis->decodeBoard(*(IS_Iterator::operator*()));
                };
                
            protected:
                ChessInformationSet* cis;
        };
        
        CIS_Iterator begin() noexcept
        {
            return CIS_Iterator(this);
        };
        
        CIS_Iterator end() noexcept
        {
            return CIS_Iterator(this,this->size());
        };
        
    private:
        std::queue<std::uint64_t> incompatibleBoards;        
};
}
#endif // INFORMATIONSET_H
