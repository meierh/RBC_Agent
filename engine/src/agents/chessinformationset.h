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

#include <gtest/gtest.h>
#include <iostream>
#include <cassert>
#include <memory>
#include <cstdint>
#include <queue>
#include <limits>
#include "informationset.h"
#include <functional>

#include "../environments/open_spiel/openspielstate.h"

namespace crazyara {
    
constexpr std::uint64_t chessInfoSize= 6*64 + // pieces {pawn,knight,bishop,rook,queen,king}
                                       2*1 +  // castling legal {kingside, queenside}
                                       1*64 + // en-passent allowed
                                       7 +  // number of moves for 50-moves rule
                                       7; // probability

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
                Square();
                Square(ChessInformationSet::ChessColumn column,ChessInformationSet::ChessRow row);
                Square(const open_spiel::chess_common::Square& os_sq);
                bool operator==(const Square& other) const
                {
                    return column==other.column && row==other.row;
                }
                bool operator!=(const Square& other) const {return !operator==(other);}
                typedef struct{
                    auto operator()(const Square &s) const -> size_t {
                        return std::hash<ChessColumn>{}(s.column)^std::hash<ChessRow>{}(s.row);
                    }
                }Hasher;
                
                // general movement vectors
                bool vertPlus(std::uint8_t multiple);
                bool vertMinus(std::uint8_t multiple);
                bool horizPlus(std::uint8_t multiple);
                bool horizMinus(std::uint8_t multiple);
                bool diagVertPlusHorizPlus(std::uint8_t multiple);
                bool diagVertMinusHorizPlus(std::uint8_t multiple);
                bool diagVertPlusHorizMinus(std::uint8_t multiple);
                bool diagVertMinusHorizMinus(std::uint8_t multiple);
                
                // special knight moves
                /*
                bool knightVertPlusHorizPlus();
                bool knightVertPlusHorizMinus();
                
                bool knightVertMinusHorizPlus();
                bool knightVertMinusHorizMinus();
                
                bool knightHorizPlusVertPlus();
                bool knightHorizPlusVertMinus();
                
                bool knightHorizMinusVertPlus();
                bool knightHorizMinusVertMinus();
                */
                
                bool validSquare(std::int8_t column, std::int8_t row);

                std::string to_string() const
                {
                    char col = 'a' + static_cast<uint>(column);
                    char row = '1' + static_cast<uint>(row);
                    return std::string() + col + row;
                };
                
            private:
                bool moveSquare(std::int8_t deltaCol, std::int8_t deltaRow);                
        };
        enum class Piece {pawn1=0,pawn2=1,pawn3=2,pawn4=3,pawn5=4,pawn6=5,pawn7=6,pawn8=7,
                          rook1=8,knight1=9,bishop1=10,queen=11,king=12,bishop2=13,knight2=14,rook2=15};
        enum class PieceType{pawn=0,knight=1,bishop=2,rook=3,queen=4,king=5,empty=6};
        static PieceType OpenSpielPieceType_to_CISPieceType(const open_spiel::chess::PieceType os_pT);
        static open_spiel::chess::PieceType CISPieceType_to_OpenSpielPieceType(const PieceType cis_pT);

        static Square boardIndexToSquare(std::uint8_t index)
        {
            std::uint8_t x = index / 8; //column {A-H}
            std::uint8_t y = index % 8; //row {1-8}
            ChessInformationSet::Square sq;
            sq.column = static_cast<ChessInformationSet::ChessColumn>(x);
            sq.row = static_cast<ChessInformationSet::ChessRow>(y);
            return sq;
        };
        
        static std::uint8_t squareToBoardIndex(Square sq)
        {
            std::uint8_t x = static_cast<std::uint8_t>(sq.column); //column {A-H}
            std::uint8_t y = static_cast<std::uint8_t>(sq.row); //row {1-8}
            return x*8+y;
        };
        
        class OnePlayerChessInfo
        {
            public:
                // list of all squares with pieces
                std::vector<ChessInformationSet::Square> pawns;
                std::vector<ChessInformationSet::Square> knights;
                std::vector<ChessInformationSet::Square> bishops;
                std::vector<ChessInformationSet::Square> rooks;
                std::vector<ChessInformationSet::Square> queens;
                std::vector<ChessInformationSet::Square> kings;
                
                // castling legal
                bool kingside=true;
                bool queenside=true;
                
                // en-passent legal
                std::vector<ChessInformationSet::Square> en_passant;
                
                // fifty move rule counter
                std::uint8_t no_progress_count=0;

                std::function<bool(const ChessInformationSet::Square&)> getBlockCheck();
                std::function<bool(const ChessInformationSet::Square&)> getBlockCheck
                (
                    const std::vector<ChessInformationSet::Square>& onePieceType,
                    const PieceType pT
                );                
                std::function<std::pair<bool,PieceType>(const ChessInformationSet::Square&)> getSquarePieceTypeCheck();
                std::function<std::vector<ChessInformationSet::Square>::iterator(const ChessInformationSet::Square&)> getPieceIter
                (
                    std::vector<ChessInformationSet::Square>& onePieceType
                );
                
                bool operator==(const OnePlayerChessInfo& other) const
                {
                    using CIS = ChessInformationSet;
                    bool equal=true;
                    auto compareVectors = []
                    (
                        const std::vector<CIS::Square>& a,
                        const std::vector<CIS::Square>& b
                    )
                    {
                        std::unordered_set<CIS::Square,CIS::Square::Hasher> a_set(a.begin(),a.end());
                        for(const CIS::Square& sq : b)
                        {
                            auto iter = a_set.find(sq);
                            if(iter!=a_set.end())
                                a_set.erase(iter);
                            else
                                return false;
                        }
                        return a_set.empty();
                    };
                    equal &= compareVectors(pawns,other.pawns);
                    equal &= compareVectors(knights,other.knights);
                    equal &= compareVectors(bishops,other.bishops);
                    equal &= compareVectors(rooks,other.rooks);
                    equal &= compareVectors(queens,other.queens);
                    equal &= compareVectors(kings,other.kings);
                    equal &= kingside==other.kingside;
                    equal &= queenside==other.queenside;
                    equal &= compareVectors(en_passant,other.en_passant);
                    equal &= no_progress_count==other.no_progress_count;
                    return equal;
                };
                
                std::string to_string() const
                {
                    auto printPieces = [](const std::vector<ChessInformationSet::Square>& list,std::string name)
                    {
                        std::string res = name+"( ";
                        for(ChessInformationSet::Square sq : list)
                            res+=sq.to_string()+" ";
                        res+=") ";
                        return res;
                    };
                    std::string res;
                    res += printPieces(pawns,"p");
                    res += printPieces(knights,"n");
                    res += printPieces(bishops,"b");
                    res += printPieces(rooks,"r");
                    res += printPieces(queens,"q");
                    res += printPieces(kings,"k");
                    res += "Ck:"+std::to_string(kingside);
                    res += " Cq:"+std::to_string(queenside)+" ";
                    res += printPieces(en_passant,"enPass");
                    res += std::to_string(no_progress_count);
                    return res;
                };
                friend std::ostream& operator<<(std::ostream&os, const OnePlayerChessInfo& opci)
                {
                    os<<opci.to_string();
                    return os;
                }
                
            private:
                std::unordered_map<Square,PieceType,Square::Hasher> squareToPieceMap;
                std::unordered_map<Square,std::vector<ChessInformationSet::Square>::iterator,Square::Hasher> squareToPieceIter;
                
                FRIEND_TEST(chessinformationsetoneplayerinfo_test, lambdafunctions_test);
        };
        
        class BoardClause
        // Class for logical clause consisting of multiple literals evaluated as a disjunction
        {
            public:
                enum class PieceType {pawn=0,knight=1,bishop=2,rook=3,queen=4,king=5,any=6,none=7};
            private:
                unsigned int literalNbr;
                std::vector<Square> boardPlaces;
                std::vector<PieceType> boardPlaceTypes;
                std::vector<bool> conditionBool;
            public:
                BoardClause(Square boardPlace,PieceType boardPlaceType)
                {
                    boardPlaces.push_back(boardPlace);
                    boardPlaceTypes.push_back(boardPlaceType);
                    conditionBool.push_back(true);
                    literalNbr = 1;
                };

                BoardClause operator|(const BoardClause& rhs) const
                {
                    BoardClause lhs = *this;
                    lhs.boardPlaces.insert
                        (lhs.boardPlaces.end(),rhs.boardPlaces.begin(),rhs.boardPlaces.end());
                    lhs.boardPlaceTypes.insert
                        (lhs.boardPlaceTypes.end(),rhs.boardPlaceTypes.begin(),rhs.boardPlaceTypes.end());
                    lhs.conditionBool.insert
                        (lhs.conditionBool.end(),rhs.conditionBool.begin(),rhs.conditionBool.end());
                    lhs.literalNbr += rhs.literalNbr;
                    return lhs;
                };
                
                BoardClause operator!() const
                {
                    BoardClause lhs = *this;
                    for(unsigned int i=0; i<lhs.literalNbr; i++)
                        lhs.conditionBool[i] = !lhs.conditionBool[i];
                    return lhs;
                };
                
                bool operator()(OnePlayerChessInfo& info) const
                // one condition must be true for the Board Condition to evaluate true
                {
                    std::function<std::pair<bool,ChessInformationSet::PieceType>(Square)> spPieceTypeCheck = info.getSquarePieceTypeCheck();
                    bool oneTrue = false;
                    
                    for(unsigned int clauseInd=0; clauseInd<literalNbr; clauseInd++)
                    {
                        const Square& sq = boardPlaces[clauseInd];
                        const PieceType& pT = boardPlaceTypes[clauseInd];
                        bool boolVal = conditionBool[clauseInd];
                        std::pair<bool,ChessInformationSet::PieceType> sqPieceType = spPieceTypeCheck(sq);
                        bool thisClauseBool;
                        if(pT == PieceType::none)
                        {
                            thisClauseBool =(sqPieceType.first==false);
                        }
                        else if(pT == PieceType::any)
                        {
                            thisClauseBool = (sqPieceType.first==true);
                        }
                        else
                        {
                            if(sqPieceType.first)
                            {
                                unsigned int pT_int_given = static_cast<unsigned int>(sqPieceType.second);
                                unsigned int pT_int_required = static_cast<unsigned int>(pT);
                                thisClauseBool = (pT_int_given==pT_int_required);
                            }
                            else
                                thisClauseBool = false;
                        }
                        if(!boolVal)
                            thisClauseBool = !thisClauseBool;
                        oneTrue = oneTrue || thisClauseBool;
                    }
                    return oneTrue;
                };
                
                FRIEND_TEST(chessinformationsetboardclause_test, constructor_test);
                FRIEND_TEST(chessinformationsetboardclause_test, orOperator_test);
                FRIEND_TEST(chessinformationsetboardclause_test, notOperator_test);
                FRIEND_TEST(chessinformationsetboardclause_test, evalOperator_test);
        };
        
        /**
         * Marks boards incompatible with observations
         * @param noPieces
         * @param unknownPieces
         * @param knownPieces
         */
        void markIncompatibleBoards(const std::vector<BoardClause>& conditions);
        
        void add(OnePlayerChessInfo& item, double probability);
        
        void add(std::vector<std::pair<OnePlayerChessInfo,double>>& items);
        
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
        void setBoard(OnePlayerChessInfo& pieces, double probability, std::uint64_t index);
        
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
        std::unique_ptr<std::pair<OnePlayerChessInfo,double>> getBoard(const std::uint64_t index) const;
        
        std::unique_ptr<std::pair<OnePlayerChessInfo,double>> decodeBoard
        (
            const std::bitset<chessInfoSize>& bits
        ) const;
        
        std::unique_ptr<std::bitset<chessInfoSize>> encodeBoard
        (
            OnePlayerChessInfo& piecesInfo,
            double probability
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

                std::unique_ptr<std::pair<OnePlayerChessInfo,double>> operator*() const noexcept
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
        
        FRIEND_TEST(chessinformationsetsquare_test, constructorAndEqual_test);
        FRIEND_TEST(chessinformationsetsquare_test, generalmovement_test);
        FRIEND_TEST(chessinformationsetsquare_test, validSquare_test);
        FRIEND_TEST(chessinformationset_test, encodeDecode_test);
        FRIEND_TEST(chessinformationset_test, addSetAndGetBoards_test);
};
}
#endif // INFORMATIONSET_H