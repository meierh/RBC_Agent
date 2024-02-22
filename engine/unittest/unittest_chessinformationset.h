#include <iostream>
#include "chessinformationset.h"
#include "rbcagent.h"

auto os_SquareToIndex = open_spiel::chess::SquareToIndex;
using os_Square = open_spiel::chess::Square;

namespace crazyara {
TEST(chessinformationsetsquare_test, constructorAndEqual_test)
{
    using CIS = ChessInformationSet;
    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    CIS::Square sq3(CIS::ChessColumn::E,CIS::ChessRow::one);
    CIS::Square sq4(CIS::ChessColumn::B,CIS::ChessRow::three);
    EXPECT_EQ(sq1,sq1);
    EXPECT_NE(sq2,sq1);
    EXPECT_NE(sq2,sq1);
    EXPECT_EQ(sq1,sq4);
}

TEST(chessinformationsetsquare_test, toString_test)
{
    using CIS = ChessInformationSet;
    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    CIS::Square sq3(CIS::ChessColumn::E,CIS::ChessRow::one);
    CIS::Square sq4(CIS::ChessColumn::B,CIS::ChessRow::three);
    EXPECT_EQ(sq1.to_string(),"b3");
    EXPECT_EQ(sq2.to_string(),"a7");
    EXPECT_EQ(sq3.to_string(),"e1");
    EXPECT_EQ(sq4.to_string(),"b3");
}

TEST(chessinformationsetsquare_test, generalmovement_test)
{
    using CIS = ChessInformationSet;
    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    
    EXPECT_EQ(sq1.vertPlus(10),false);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::three));
    EXPECT_EQ(sq1.vertMinus(10),false);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::three));
    EXPECT_EQ(sq1.horizPlus(10),false);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::three));
    EXPECT_EQ(sq1.horizMinus(10),false);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::three));
    EXPECT_EQ(sq1.diagVertPlusHorizPlus(10),false);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::three));
    EXPECT_EQ(sq1.diagVertMinusHorizPlus(10),false);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::three));
    EXPECT_EQ(sq1.diagVertPlusHorizMinus(10),false);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::three));
    EXPECT_EQ(sq1.diagVertMinusHorizMinus(10),false);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::three));
    
    sq1 = CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::four);
    EXPECT_EQ(sq1.vertPlus(2),true);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::six));
    sq1 = CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::four);
    EXPECT_EQ(sq1.vertMinus(2),true);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::two));
    sq1 = CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::four);
    EXPECT_EQ(sq1.horizPlus(2),true);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::F,CIS::ChessRow::four));
    sq1 = CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::four);
    EXPECT_EQ(sq1.horizMinus(2),true);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::four));
    sq1 = CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::four);
    EXPECT_EQ(sq1.diagVertPlusHorizPlus(2),true);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::F,CIS::ChessRow::six));
    sq1 = CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::four);
    EXPECT_EQ(sq1.diagVertMinusHorizPlus(2),true);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::F,CIS::ChessRow::two));
    sq1 = CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::four);
    EXPECT_EQ(sq1.diagVertPlusHorizMinus(2),true);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::six));
    sq1 = CIS::Square(CIS::ChessColumn::D,CIS::ChessRow::four);
    EXPECT_EQ(sq1.diagVertMinusHorizMinus(2),true);
    EXPECT_EQ(sq1,CIS::Square(CIS::ChessColumn::B,CIS::ChessRow::two));
}

TEST(chessinformationsetsquare_test, validSquare_test)
{
    using CIS = ChessInformationSet;
    CIS::Square sq;
    
    EXPECT_EQ(sq.validSquare(-1,-2),false);
    EXPECT_EQ(sq.validSquare(1,2),true);
    EXPECT_EQ(sq.validSquare(1,9),false);
    EXPECT_EQ(sq.validSquare(1,8),false);
    EXPECT_EQ(sq.validSquare(0,7),true);
}

TEST(chessinformationsetsquare_test, boardAndIndex_test)
{
    using CIS = ChessInformationSet;
    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    CIS::Square sq3(CIS::ChessColumn::E,CIS::ChessRow::one);
    CIS::Square sq4(CIS::ChessColumn::H,CIS::ChessRow::six);
    CIS::Square sq5(CIS::ChessColumn::B,CIS::ChessRow::seven);
    
    EXPECT_EQ(CIS::boardIndexToSquare(CIS::squareToBoardIndex(sq1)),sq1);
    EXPECT_EQ(CIS::boardIndexToSquare(CIS::squareToBoardIndex(sq2)),sq2);
    EXPECT_EQ(CIS::boardIndexToSquare(CIS::squareToBoardIndex(sq3)),sq3);
    EXPECT_EQ(CIS::boardIndexToSquare(CIS::squareToBoardIndex(sq4)),sq4);
    EXPECT_EQ(CIS::boardIndexToSquare(CIS::squareToBoardIndex(sq5)),sq5);
    
    EXPECT_EQ(CIS::squareToBoardIndex(sq1),os_SquareToIndex(os_Square{1,2}, 8));
    EXPECT_EQ(CIS::squareToBoardIndex(sq2),os_SquareToIndex(os_Square{0,6}, 8));
    EXPECT_EQ(CIS::squareToBoardIndex(sq3),os_SquareToIndex(os_Square{4,0}, 8));
    EXPECT_EQ(CIS::squareToBoardIndex(sq4),os_SquareToIndex(os_Square{7,5}, 8));
    EXPECT_EQ(CIS::squareToBoardIndex(sq5),os_SquareToIndex(os_Square{1,6}, 8));
    
    EXPECT_EQ(CIS::squareToBoardIndex(sq1),17);
    EXPECT_EQ(CIS::squareToBoardIndex(sq2),48);
    EXPECT_EQ(CIS::squareToBoardIndex(sq3),4);
    EXPECT_EQ(CIS::squareToBoardIndex(sq4),47);
    EXPECT_EQ(CIS::squareToBoardIndex(sq5),49);
}

TEST(chessinformationsetoneplayerinfo_test, lambdafunctions_test)
{
    using CIS = ChessInformationSet;
    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    CIS::Square sq3(CIS::ChessColumn::E,CIS::ChessRow::one);
    CIS::Square sq4(CIS::ChessColumn::B,CIS::ChessRow::two);
    CIS::Square sq5(CIS::ChessColumn::C,CIS::ChessRow::five);
    CIS::Square sq6(CIS::ChessColumn::F,CIS::ChessRow::four);
    CIS::Square sq7(CIS::ChessColumn::E,CIS::ChessRow::eight);
    CIS::Square sq8(CIS::ChessColumn::D,CIS::ChessRow::seven);
    CIS::Square sq9(CIS::ChessColumn::D,CIS::ChessRow::one);
    CIS::Square sq10(CIS::ChessColumn::E,CIS::ChessRow::five);
    CIS::Square sq11(CIS::ChessColumn::C,CIS::ChessRow::seven);
    
    CIS::Square sq12(CIS::ChessColumn::G,CIS::ChessRow::one);
    CIS::Square sq13(CIS::ChessColumn::G,CIS::ChessRow::eight);
    CIS::Square sq14(CIS::ChessColumn::H,CIS::ChessRow::one);
    CIS::Square sq15(CIS::ChessColumn::H,CIS::ChessRow::five);
    CIS::Square sq16(CIS::ChessColumn::A,CIS::ChessRow::eight);

    CIS::OnePlayerChessInfo opci;
    opci.pawns = {{sq1,sq2,sq3}};
    opci.knights = {{sq4,sq5}};
    opci.bishops = {{sq6,sq7}};
    opci.rooks = {{sq8,sq9}};
    opci.queens = {{sq10}};
    opci.kings =  {{sq11}};
    
    auto blockCheck = opci.getBlockCheck();
    EXPECT_EQ(blockCheck(sq1),true);
    EXPECT_EQ(blockCheck(sq2),true);
    EXPECT_EQ(blockCheck(sq3),true);
    EXPECT_EQ(blockCheck(sq4),true);
    EXPECT_EQ(blockCheck(sq5),true);
    EXPECT_EQ(blockCheck(sq6),true);
    EXPECT_EQ(blockCheck(sq7),true);
    EXPECT_EQ(blockCheck(sq8),true);
    EXPECT_EQ(blockCheck(sq9),true);
    EXPECT_EQ(blockCheck(sq10),true);
    EXPECT_EQ(blockCheck(sq11),true);    
    EXPECT_EQ(blockCheck(sq12),false);
    EXPECT_EQ(blockCheck(sq13),false);
    EXPECT_EQ(blockCheck(sq14),false);
    EXPECT_EQ(blockCheck(sq15),false);
    EXPECT_EQ(blockCheck(sq16),false);
    
    auto pawnBlockCheck = opci.getBlockCheck(opci.pawns,CIS::PieceType::pawn);
    EXPECT_EQ(pawnBlockCheck(sq1),true);
    EXPECT_EQ(pawnBlockCheck(sq2),true);
    EXPECT_EQ(pawnBlockCheck(sq3),true); 
    EXPECT_EQ(pawnBlockCheck(sq12),false);
    EXPECT_EQ(pawnBlockCheck(sq13),false);
    EXPECT_EQ(pawnBlockCheck(sq14),false);
    EXPECT_EQ(pawnBlockCheck(sq15),false);
    EXPECT_EQ(pawnBlockCheck(sq16),false);    
    
    auto knightBlockCheck = opci.getBlockCheck(opci.knights,CIS::PieceType::knight);
    EXPECT_EQ(knightBlockCheck(sq4),true);
    EXPECT_EQ(knightBlockCheck(sq5),true);
    EXPECT_EQ(knightBlockCheck(sq12),false);
    EXPECT_EQ(knightBlockCheck(sq13),false);
    EXPECT_EQ(knightBlockCheck(sq14),false);
    EXPECT_EQ(knightBlockCheck(sq15),false);
    EXPECT_EQ(knightBlockCheck(sq16),false);
    
    auto bishopBlockCheck = opci.getBlockCheck(opci.bishops,CIS::PieceType::bishop);
    EXPECT_EQ(bishopBlockCheck(sq6),true);
    EXPECT_EQ(bishopBlockCheck(sq7),true);
    EXPECT_EQ(bishopBlockCheck(sq12),false);
    EXPECT_EQ(bishopBlockCheck(sq13),false);
    EXPECT_EQ(bishopBlockCheck(sq14),false);
    EXPECT_EQ(bishopBlockCheck(sq15),false);
    EXPECT_EQ(bishopBlockCheck(sq16),false);
    
    auto rookBlockCheck = opci.getBlockCheck(opci.rooks,CIS::PieceType::rook);
    EXPECT_EQ(rookBlockCheck(sq8),true);
    EXPECT_EQ(rookBlockCheck(sq9),true);
    EXPECT_EQ(rookBlockCheck(sq12),false);
    EXPECT_EQ(rookBlockCheck(sq13),false);
    EXPECT_EQ(rookBlockCheck(sq14),false);
    EXPECT_EQ(rookBlockCheck(sq15),false);
    EXPECT_EQ(rookBlockCheck(sq16),false);
    
    auto queenBlockCheck = opci.getBlockCheck(opci.queens,CIS::PieceType::queen);
    EXPECT_EQ(queenBlockCheck(sq10),true);
    EXPECT_EQ(queenBlockCheck(sq12),false);
    EXPECT_EQ(queenBlockCheck(sq13),false);
    EXPECT_EQ(queenBlockCheck(sq14),false);
    EXPECT_EQ(queenBlockCheck(sq15),false);
    EXPECT_EQ(queenBlockCheck(sq16),false);
    
    auto kingBlockCheck = opci.getBlockCheck(opci.kings,CIS::PieceType::king);
    EXPECT_EQ(kingBlockCheck(sq11),true);
    EXPECT_EQ(kingBlockCheck(sq12),false);
    EXPECT_EQ(kingBlockCheck(sq13),false);
    EXPECT_EQ(kingBlockCheck(sq14),false);
    EXPECT_EQ(kingBlockCheck(sq15),false);
    EXPECT_EQ(kingBlockCheck(sq16),false);
    
    auto squarePieceTypeCheck = opci.getSquarePieceTypeCheck();
    EXPECT_EQ(squarePieceTypeCheck(sq1),std::make_pair<>(true,CIS::PieceType::pawn));
    EXPECT_EQ(squarePieceTypeCheck(sq2),std::make_pair<>(true,CIS::PieceType::pawn));
    EXPECT_EQ(squarePieceTypeCheck(sq3),std::make_pair<>(true,CIS::PieceType::pawn));
    EXPECT_EQ(squarePieceTypeCheck(sq4),std::make_pair<>(true,CIS::PieceType::knight));
    EXPECT_EQ(squarePieceTypeCheck(sq5),std::make_pair<>(true,CIS::PieceType::knight));
    EXPECT_EQ(squarePieceTypeCheck(sq6),std::make_pair<>(true,CIS::PieceType::bishop));
    EXPECT_EQ(squarePieceTypeCheck(sq7),std::make_pair<>(true,CIS::PieceType::bishop));
    EXPECT_EQ(squarePieceTypeCheck(sq8),std::make_pair<>(true,CIS::PieceType::rook));
    EXPECT_EQ(squarePieceTypeCheck(sq9),std::make_pair<>(true,CIS::PieceType::rook));
    EXPECT_EQ(squarePieceTypeCheck(sq10),std::make_pair<>(true,CIS::PieceType::queen));
    EXPECT_EQ(squarePieceTypeCheck(sq11),std::make_pair<>(true,CIS::PieceType::king));  
    EXPECT_EQ(squarePieceTypeCheck(sq12),std::make_pair<>(false,CIS::PieceType::pawn));
    EXPECT_EQ(squarePieceTypeCheck(sq13),std::make_pair<>(false,CIS::PieceType::pawn));
    EXPECT_EQ(squarePieceTypeCheck(sq14),std::make_pair<>(false,CIS::PieceType::pawn));
    EXPECT_EQ(squarePieceTypeCheck(sq15),std::make_pair<>(false,CIS::PieceType::pawn));
    EXPECT_EQ(squarePieceTypeCheck(sq16),std::make_pair<>(false,CIS::PieceType::pawn));
    
    auto squareToPawnPieceIter = opci.getPieceIter(opci.pawns);
    EXPECT_EQ(squareToPawnPieceIter(sq1),opci.pawns.begin()+0);
    EXPECT_EQ(squareToPawnPieceIter(sq2),opci.pawns.begin()+1);
    EXPECT_EQ(squareToPawnPieceIter(sq3),opci.pawns.begin()+2);
    EXPECT_EQ(squareToPawnPieceIter(sq12),opci.pawns.end());
    EXPECT_EQ(squareToPawnPieceIter(sq13),opci.pawns.end());
    EXPECT_EQ(squareToPawnPieceIter(sq14),opci.pawns.end());
    EXPECT_EQ(squareToPawnPieceIter(sq15),opci.pawns.end());
    EXPECT_EQ(squareToPawnPieceIter(sq16),opci.pawns.end()); 

    auto squareToKnightPieceIter = opci.getPieceIter(opci.knights);
    EXPECT_EQ(squareToKnightPieceIter(sq4),opci.knights.begin()+0);
    EXPECT_EQ(squareToKnightPieceIter(sq5),opci.knights.begin()+1);
    EXPECT_EQ(squareToKnightPieceIter(sq12),opci.knights.end());
    EXPECT_EQ(squareToKnightPieceIter(sq13),opci.knights.end());
    EXPECT_EQ(squareToKnightPieceIter(sq14),opci.knights.end());
    EXPECT_EQ(squareToKnightPieceIter(sq15),opci.knights.end());
    EXPECT_EQ(squareToKnightPieceIter(sq16),opci.knights.end()); 
    
    auto squareToBishopPieceIter = opci.getPieceIter(opci.bishops);
    EXPECT_EQ(squareToBishopPieceIter(sq6),opci.bishops.begin()+0);
    EXPECT_EQ(squareToBishopPieceIter(sq7),opci.bishops.begin()+1);
    EXPECT_EQ(squareToBishopPieceIter(sq12),opci.bishops.end());
    EXPECT_EQ(squareToBishopPieceIter(sq13),opci.bishops.end());
    EXPECT_EQ(squareToBishopPieceIter(sq14),opci.bishops.end());
    EXPECT_EQ(squareToBishopPieceIter(sq15),opci.bishops.end());
    EXPECT_EQ(squareToBishopPieceIter(sq16),opci.bishops.end()); 
    
    auto squareToRookPieceIter = opci.getPieceIter(opci.rooks);
    EXPECT_EQ(squareToRookPieceIter(sq8),opci.rooks.begin()+0);
    EXPECT_EQ(squareToRookPieceIter(sq9),opci.rooks.begin()+1);
    EXPECT_EQ(squareToRookPieceIter(sq12),opci.rooks.end());
    EXPECT_EQ(squareToRookPieceIter(sq13),opci.rooks.end());
    EXPECT_EQ(squareToRookPieceIter(sq14),opci.rooks.end());
    EXPECT_EQ(squareToRookPieceIter(sq15),opci.rooks.end());
    EXPECT_EQ(squareToRookPieceIter(sq16),opci.rooks.end()); 
    
    auto squareToQueenPieceIter = opci.getPieceIter(opci.queens);
    EXPECT_EQ(squareToQueenPieceIter(sq10),opci.queens.begin()+0);
    EXPECT_EQ(squareToQueenPieceIter(sq12),opci.queens.end());
    EXPECT_EQ(squareToQueenPieceIter(sq13),opci.queens.end());
    EXPECT_EQ(squareToQueenPieceIter(sq14),opci.queens.end());
    EXPECT_EQ(squareToQueenPieceIter(sq15),opci.queens.end());
    EXPECT_EQ(squareToQueenPieceIter(sq16),opci.queens.end());
    
    auto squareToKingPieceIter = opci.getPieceIter(opci.kings);
    EXPECT_EQ(squareToKingPieceIter(sq11),opci.kings.begin()+0);
    EXPECT_EQ(squareToKingPieceIter(sq12),opci.kings.end());
    EXPECT_EQ(squareToKingPieceIter(sq13),opci.kings.end());
    EXPECT_EQ(squareToKingPieceIter(sq14),opci.kings.end());
    EXPECT_EQ(squareToKingPieceIter(sq15),opci.kings.end());
    EXPECT_EQ(squareToKingPieceIter(sq16),opci.kings.end()); 
}

TEST(chessinformationsetboardclause_test, constructor_test)
{
    using CIS = ChessInformationSet;

    CIS::BoardClause bc1 (CIS::Square(),CIS::BoardClause::PieceType::pawn);    
    EXPECT_EQ(bc1.literalNbr,1);
    EXPECT_EQ(bc1.boardPlaces.size(),1);
    EXPECT_EQ(bc1.boardPlaceTypes.size(),1);
    EXPECT_EQ(bc1.conditionBool.size(),1);
}

TEST(chessinformationsetboardclause_test, orOperator_test)
{
    using CIS = ChessInformationSet;

    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    
    CIS::BoardClause bc1 (sq1,CIS::BoardClause::PieceType::pawn);
    CIS::BoardClause bc2 (sq2,CIS::BoardClause::PieceType::none);    
    bc1 = bc1 | bc2;
    
    EXPECT_EQ(bc1.literalNbr,2);
    EXPECT_EQ(bc1.boardPlaces.size(),2);
    EXPECT_EQ(bc1.boardPlaces[0],sq1);
    EXPECT_EQ(bc1.boardPlaces[1],sq2);
    EXPECT_EQ(bc1.boardPlaceTypes.size(),2);
    EXPECT_EQ(bc1.boardPlaceTypes[0],CIS::BoardClause::PieceType::pawn);
    EXPECT_EQ(bc1.boardPlaceTypes[1],CIS::BoardClause::PieceType::none);
    EXPECT_EQ(bc1.conditionBool.size(),2);
    EXPECT_EQ(bc1.conditionBool[0],true);
    EXPECT_EQ(bc1.conditionBool[1],true);
}

TEST(chessinformationsetboardclause_test, notOperator_test)
{
    using CIS = ChessInformationSet;

    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    
    CIS::BoardClause bc1 (sq1,CIS::BoardClause::PieceType::pawn);
    bc1 = !bc1;
    CIS::BoardClause bc2 (sq2,CIS::BoardClause::PieceType::none);    
    bc1 = bc1 | bc2;
    
    EXPECT_EQ(bc1.literalNbr,2);
    EXPECT_EQ(bc1.boardPlaces.size(),2);
    EXPECT_EQ(bc1.boardPlaces[0],sq1);
    EXPECT_EQ(bc1.boardPlaces[1],sq2);
    EXPECT_EQ(bc1.boardPlaceTypes.size(),2);
    EXPECT_EQ(bc1.boardPlaceTypes[0],CIS::BoardClause::PieceType::pawn);
    EXPECT_EQ(bc1.boardPlaceTypes[1],CIS::BoardClause::PieceType::none);
    EXPECT_EQ(bc1.conditionBool.size(),2);
    EXPECT_EQ(bc1.conditionBool[0],false);
    EXPECT_EQ(bc1.conditionBool[1],true);
    
    bc1 = !bc1;
    EXPECT_EQ(bc1.conditionBool[0],true);
    EXPECT_EQ(bc1.conditionBool[1],false);
}

TEST(chessinformationsetboardclause_test, evalOperator_test)
{
    using CIS = ChessInformationSet;
    using PT = CIS::BoardClause::PieceType;
    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    CIS::Square sq3(CIS::ChessColumn::E,CIS::ChessRow::one);
    CIS::Square sq4(CIS::ChessColumn::B,CIS::ChessRow::two);
    CIS::Square sq5(CIS::ChessColumn::C,CIS::ChessRow::five);
    CIS::Square sq6(CIS::ChessColumn::F,CIS::ChessRow::four);
    CIS::Square sq7(CIS::ChessColumn::E,CIS::ChessRow::eight);
    CIS::Square sq8(CIS::ChessColumn::D,CIS::ChessRow::seven);
    CIS::Square sq9(CIS::ChessColumn::D,CIS::ChessRow::one);
    CIS::Square sq10(CIS::ChessColumn::E,CIS::ChessRow::five);
    CIS::Square sq11(CIS::ChessColumn::C,CIS::ChessRow::seven);
    
    CIS::Square sq12(CIS::ChessColumn::G,CIS::ChessRow::one);
    CIS::Square sq13(CIS::ChessColumn::G,CIS::ChessRow::eight);
    CIS::Square sq14(CIS::ChessColumn::H,CIS::ChessRow::one);
    CIS::Square sq15(CIS::ChessColumn::H,CIS::ChessRow::five);
    CIS::Square sq16(CIS::ChessColumn::A,CIS::ChessRow::eight);

    CIS::OnePlayerChessInfo opci;
    opci.pawns = {{sq1,sq2,sq3}};
    opci.knights = {{sq4,sq5}};
    opci.bishops = {{sq6,sq7}};
    opci.rooks = {{sq8,sq9}};
    opci.queens = {{sq10}};
    opci.kings =  {{sq11}};
    
    CIS::BoardClause bc1 (sq1,PT::pawn);
    CIS::BoardClause bc2=!CIS::BoardClause(sq2,PT::none);
    CIS::BoardClause bc3 (sq6,PT::bishop);
    CIS::BoardClause bc4 (sq10,PT::queen);
    CIS::BoardClause bc5 (sq12,PT::none);
    CIS::BoardClause bcFull1 = bc1 | bc2 | bc3 | bc4 | bc5;    
    EXPECT_EQ(bc1(opci),true);
    EXPECT_EQ(bc2(opci),true);
    EXPECT_EQ(bc3(opci),true);
    EXPECT_EQ(bc4(opci),true);
    EXPECT_EQ(bc5(opci),true);
    EXPECT_EQ(bcFull1(opci),true);
    
    CIS::BoardClause bc6 (sq1,PT::pawn);
    CIS::BoardClause bc7 (sq2,PT::none);
    CIS::BoardClause bc8 (sq6,PT::bishop);
    CIS::BoardClause bc9 (sq10,PT::none);
    CIS::BoardClause bc10 (sq12,PT::none);
    CIS::BoardClause bcFull2 = bc6 | bc7 | bc8 | bc9 | bc10;    
    EXPECT_EQ(bc6(opci),true);
    EXPECT_EQ(bc7(opci),false);
    EXPECT_EQ(bc8(opci),true);
    EXPECT_EQ(bc9(opci),false);
    EXPECT_EQ(bc10(opci),true);
    EXPECT_EQ(bcFull2(opci),true);
    
    CIS::BoardClause bcFull3 = !bc6 | bc7 | !bc8 | bc9 | !bc10;
    EXPECT_EQ(bcFull3(opci),false);
}

TEST(chessinformationset_test, encodeDecode_test)
{
    using CIS = ChessInformationSet;
    CIS cis;
    using PT = CIS::BoardClause::PieceType;
    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    CIS::Square sq3(CIS::ChessColumn::E,CIS::ChessRow::one);
    CIS::Square sq4(CIS::ChessColumn::B,CIS::ChessRow::two);
    CIS::Square sq5(CIS::ChessColumn::C,CIS::ChessRow::five);
    CIS::Square sq6(CIS::ChessColumn::F,CIS::ChessRow::four);
    CIS::Square sq7(CIS::ChessColumn::E,CIS::ChessRow::eight);
    CIS::Square sq8(CIS::ChessColumn::D,CIS::ChessRow::seven);
    CIS::Square sq9(CIS::ChessColumn::D,CIS::ChessRow::one);
    CIS::Square sq10(CIS::ChessColumn::E,CIS::ChessRow::five);
    CIS::Square sq11(CIS::ChessColumn::C,CIS::ChessRow::seven);
        
    CIS::OnePlayerChessInfo opci;
    opci.pawns = {{sq1,sq2,sq3}};
    opci.knights = {{sq4,sq5}};
    opci.bishops = {{sq6,sq7}};
    opci.rooks = {{sq8,sq9}};
    opci.queens = {{sq10}};
    opci.kings =  {{sq11}};
    opci.kingside=true;
    opci.queenside=false;
    opci.en_passant = {{sq2}};
    opci.no_progress_count = 7;
        
    auto encoded = cis.encodeBoard(opci,1);
    auto decoded = cis.decodeBoard(*encoded);
    CIS::OnePlayerChessInfo opci2 = decoded->first;
    EXPECT_EQ(opci,opci2);
    
    opci = CIS::OnePlayerChessInfo();        
    encoded = cis.encodeBoard(opci,1);
    decoded = cis.decodeBoard(*encoded);
    opci2 = decoded->first;
    EXPECT_EQ(opci,opci2);

    opci = CIS::OnePlayerChessInfo();
    opci.no_progress_count = 200;   
    EXPECT_ANY_THROW(cis.encodeBoard(opci,1));
    
    opci.no_progress_count = 50;   
    EXPECT_ANY_THROW(cis.encodeBoard(opci,1.1));
}

TEST(chessinformationset_test, addSetAndGetBoards_test)
{
    using CIS = ChessInformationSet;
    CIS cis;
    using PT = CIS::BoardClause::PieceType;
    CIS::Square sq1(CIS::ChessColumn::B,CIS::ChessRow::three);
    CIS::Square sq2(CIS::ChessColumn::A,CIS::ChessRow::seven);
    CIS::Square sq3(CIS::ChessColumn::E,CIS::ChessRow::one);
    CIS::Square sq4(CIS::ChessColumn::B,CIS::ChessRow::two);
    CIS::Square sq5(CIS::ChessColumn::C,CIS::ChessRow::five);
    CIS::Square sq6(CIS::ChessColumn::F,CIS::ChessRow::four);
    CIS::Square sq7(CIS::ChessColumn::E,CIS::ChessRow::eight);
    CIS::Square sq8(CIS::ChessColumn::D,CIS::ChessRow::seven);
    CIS::Square sq9(CIS::ChessColumn::D,CIS::ChessRow::one);
    CIS::Square sq10(CIS::ChessColumn::E,CIS::ChessRow::five);
    CIS::Square sq11(CIS::ChessColumn::C,CIS::ChessRow::seven);
        
    CIS::OnePlayerChessInfo opci1;
    opci1.pawns = {{sq1,sq2,sq3}};
    opci1.knights = {{sq4,sq5}};
    opci1.bishops = {{sq6,sq7}};
    opci1.rooks = {{sq8,sq9}};
    opci1.queens = {{sq10}};
    opci1.kings =  {{sq11}};
    opci1.kingside=true;
    opci1.queenside=false;
    opci1.en_passant = {{sq2}};
    opci1.no_progress_count = 7;
    
    CIS::OnePlayerChessInfo opci2;

    CIS::OnePlayerChessInfo opci3;
    opci3.no_progress_count = 50;
    opci3.kingside = true;
    opci3.queenside = false;
    
    CIS::OnePlayerChessInfo opci4;
    opci4.pawns = {{sq1,sq2,sq3}};
    opci4.bishops = {{sq6,sq7}};
    opci4.rooks = {{sq8,sq9}};
    opci4.kings =  {{sq11}};
    opci4.kingside=true;
    opci4.queenside=false;
    opci4.en_passant = {{sq2}};
    opci4.no_progress_count = 0;
    
    std::vector<std::pair<CIS::OnePlayerChessInfo,double>> boards = {{opci1,0.1},{opci2,0.3},{opci3,0.5},{opci4,0.7}};
    cis.add(boards);
    
    auto iter=cis.begin();
    for(int i=0;i<boards.size();i++,iter++)
    {
        EXPECT_EQ(boards[i].first,(*iter)->first);
    }
    cis.setBoard(opci2,0.6,3);
    EXPECT_EQ(opci2,cis.getBoard(3)->first);
    
    EXPECT_ANY_THROW(cis.setBoard(opci2,0.6,10));
}

TEST(chessinformationset_test, constructor_test)
{
    ChessInformationSet cis;
    EXPECT_EQ(cis.size(), 0);
}

TEST(chessinformationset_test, boardClause_test)
{
    using BC = ChessInformationSet::BoardClause;
    using SQ = ChessInformationSet::Square;
    using COL = ChessInformationSet::ChessColumn;
    using ROW = ChessInformationSet::ChessRow;
    using PT = ChessInformationSet::BoardClause::PieceType;
    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    RBCAgent::FullChessInfo fci1(fen1);
    BC clause11(SQ(COL::B,ROW::seven),PT::pawn);
    EXPECT_TRUE(clause11(fci1.black));
    clause11 = !clause11;
    EXPECT_FALSE(clause11(fci1.black));
    BC clause12(SQ(COL::B,ROW::eight),PT::knight);
    clause11 = clause11 | clause12;
    EXPECT_TRUE(clause11(fci1.black));
    
    std::string fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    RBCAgent::FullChessInfo fci2(fen2);
    
    std::string fen3 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2";
    RBCAgent::FullChessInfo fci3(fen3);
    
    std::string fen4 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
    RBCAgent::FullChessInfo fci4(fen4);
    
    std::string fen5 = "6r1/6pp/7r/1B5K/1P3k2/N7/3R4/8 w - - 30 79";
    RBCAgent::FullChessInfo fci5(fen5);
    
    std::string fen6 = "2k5/5P1K/3P2p1/3P2NP/p3PBp1/5B2/Q4n2/6r1 w - - 0 1";
    RBCAgent::FullChessInfo fci6(fen6);

    std::string fen7 = "5r2/p1Q3pb/5p1p/4PP2/6p1/8/pBR1Nk2/3K4 w - - 0 1";
    RBCAgent::FullChessInfo fci7(fen7);
    
    std::string fen8 = "8/5pPn/kP2b3/P1PP1N2/5p1K/r1p5/P6p/8 w - - 0 1";
    RBCAgent::FullChessInfo fci8(fen8);
    
    std::string fen9 = "3b2B1/4p3/k2p3K/2P1nq2/4bP2/6P1/2p1r1P1/7Q w - - 0 1";
    RBCAgent::FullChessInfo fci9(fen9);
    
    std::string fen10 = "2r5/2P3p1/2Q1p3/1P1p3B/2P4K/3p4/p1NP1k2/q7 w - - 0 1";
    RBCAgent::FullChessInfo fci10(fen10);
    
    std::string fen11 = "7N/8/8/8/5k2/8/1PK5/8 w - - 0 1";
    RBCAgent::FullChessInfo fci11(fen11);
    
    std::string fen12 = "8/8/8/6k1/8/K7/8/8 w - - 0 1";
    RBCAgent::FullChessInfo fci12(fen12);
    
    std::string fen13 = "2rn1b2/2pPNQ1P/K3Bpb1/2PPPpp1/np1kN1RP/p4p2/P1Pqp2R/2B4r w - - 0 1";
}

TEST(chessinformationset_test, getIncompatibleGPU_test)
{
    using BC = ChessInformationSet::BoardClause;
    using SQ = ChessInformationSet::Square;
    using COL = ChessInformationSet::ChessColumn;
    using ROW = ChessInformationSet::ChessRow;
    using PT = ChessInformationSet::BoardClause::PieceType;
    using CIS = ChessInformationSet;
    
    CIS cis0;
    std::string fen0 = "rnbqkbnr/pppppp1p/6p1/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1";
    RBCAgent::FullChessInfo fci(fen0);
    cis0.add(fci.black,1);
    BC clause03(SQ(COL::G,ROW::seven),PT::none);
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards03 = cis0.checkBoardsValidGPU({clause03});
    std::vector<std::uint8_t> compatible03 = {1};
    EXPECT_EQ(*incompBoards03,compatible03);
    
    CIS cis01;
    std::string fen01 = "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1";
    RBCAgent::FullChessInfo fci1(fen01);
    cis01.add(fci1.black,1);
    BC clause031(SQ(COL::A,ROW::two),PT::pawn);
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards031 = cis01.checkBoardsValidGPU({clause031});
    std::vector<std::uint8_t> compatible031 = {0};
    EXPECT_EQ(*incompBoards031,compatible031);
    
    CIS cis;
    std::vector<std::string> fens = 
    {
        "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/1ppppppp/8/8/8/8/pPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkb1r/pppppppp/5n2/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkb1r/1ppppppp/5n2/8/8/8/pPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppp1p/6p1/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/1ppppp1p/6p1/8/8/8/pPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkb1r/pppppp1p/5np1/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkb1r/1ppppp1p/5np1/8/8/8/pPPPPPPP/RNBQKBNR w KQkq - 0 1",

        "r1bqkbnr/pppppppp/8/2n5/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/1ppppppp/8/2n5/8/8/pPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppppppp/5n2/2n5/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/1ppppppp/5n2/2n5/8/8/pPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppp1p/6p1/2n5/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/1ppppp1p/6p1/2n5/8/8/pPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppppp1p/5np1/2n5/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/1ppppp1p/5np1/2n5/8/8/pPPPPPPP/RNBQKBNR w KQkq - 0 1"        
    };
    std::vector<std::uint8_t> compatible = {0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1};
    
    for(std::string oneFen : fens)
    {
        RBCAgent::FullChessInfo fci(oneFen);
        cis.add(fci.black,1);
    }
    
    BC clause1(SQ(COL::A,ROW::two),PT::pawn);
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards1 = cis.checkBoardsValidGPU({clause1});
    std::vector<std::uint8_t> compatible1 = {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
    EXPECT_EQ(*incompBoards1,compatible1);
    
    BC clause2(SQ(COL::F,ROW::six),PT::knight);
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards2 = cis.checkBoardsValidGPU({clause2});
    std::vector<std::uint8_t> compatible2 = {0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,};
    EXPECT_EQ(*incompBoards2,compatible2);
    
    BC clause3(SQ(COL::G,ROW::seven),PT::none);
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards3 = cis.checkBoardsValidGPU({clause3});
    std::vector<std::uint8_t> compatible3 = {0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1};
    EXPECT_EQ(*incompBoards3,compatible3);
    
    BC clause4(SQ(COL::C,ROW::five),PT::any);
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards4 = cis.checkBoardsValidGPU({clause4});
    std::vector<std::uint8_t> compatible4 = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1};
    EXPECT_EQ(*incompBoards4,compatible4);
    
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards5 = cis.checkBoardsValidGPU({clause1,clause2});
    std::vector<std::uint8_t> compatible5 = {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1};
    EXPECT_EQ(*incompBoards5,compatible5);
    
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards6 = cis.checkBoardsValidGPU({clause2 | clause3});
    std::vector<std::uint8_t> compatible6 = {0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1};
    EXPECT_EQ(*incompBoards6,compatible6);
    
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards7 = cis.checkBoardsValidGPU({clause3,clause4});
    std::vector<std::uint8_t> compatible7 = {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1};
    EXPECT_EQ(*incompBoards7,compatible7);
    
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards8 = cis.checkBoardsValidGPU({clause1,clause4});
    std::vector<std::uint8_t> compatible8 = {0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1};
    EXPECT_EQ(*incompBoards8,compatible8);
    
    std::unique_ptr<std::vector<std::uint8_t>> incompBoards9 = cis.checkBoardsValidGPU({clause1,clause2|clause3,clause4});
    std::vector<std::uint8_t> compatible9 = {0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1};
    EXPECT_EQ(*incompBoards9,compatible9);
}

TEST(chessinformationset_test, getDistributionGPU_test)
{
    using BC = ChessInformationSet::BoardClause;
    using SQ = ChessInformationSet::Square;
    using COL = ChessInformationSet::ChessColumn;
    using ROW = ChessInformationSet::ChessRow;
    using PT = ChessInformationSet::BoardClause::PieceType;
    using CIS = ChessInformationSet;
    CIS cis1;
    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    RBCAgent::FullChessInfo fci1(fen1);

    for(uint i=0; i<1000; i++)
        if(i%2==0)
            cis1.add(fci1.white,1);
        else
            cis1.add(fci1.black,1);
    
    std::unique_ptr<CIS::Distribution> incompBoards = cis1.computeDistributionGPU();
}

TEST(chessinformationset_test, getEntropyGPU_test)
{
    using BC = ChessInformationSet::BoardClause;
    using SQ = ChessInformationSet::Square;
    using COL = ChessInformationSet::ChessColumn;
    using ROW = ChessInformationSet::ChessRow;
    using PT = ChessInformationSet::BoardClause::PieceType;
    using CIS = ChessInformationSet;
    CIS cis1;
    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    RBCAgent::FullChessInfo fci1(fen1);

    /*
    for(uint i=0; i<1000; i++)
        if(i%2==0)
            cis1.add(fci1.white,1);
        else
            cis1.add(fci1.black,1);
    */
    cis1.add(fci1.white,1);
    cis1.add(fci1.black,1);

    std::unique_ptr<CIS::Distribution> incompBoards = cis1.computeDistributionGPU();
    cis1.computeEntropyGPU(*incompBoards);
    std::cout<<incompBoards->printComplete()<<std::endl;
    cis1.computeEntropyGPU(*incompBoards);
}

TEST(chessinformationset_test, getMostProbable_test)
{
    using BC = ChessInformationSet::BoardClause;
    using SQ = ChessInformationSet::Square;
    using COL = ChessInformationSet::ChessColumn;
    using ROW = ChessInformationSet::ChessRow;
    using PT = ChessInformationSet::BoardClause::PieceType;
    using CIS = ChessInformationSet;
    CIS cis1;
    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    RBCAgent::FullChessInfo fci1(fen1);

    cis1.add(fci1.white,1);
    cis1.add(fci1.white,1);
    cis1.add(fci1.black,1);
    cis1.add(fci1.white,1);
    cis1.add(fci1.black,1);
    for(int i=0; i<20000; i++)
        cis1.add(fci1.black,1);
    for(int i=0; i<30000; i++)
        cis1.add(fci1.white,1);

    std::unique_ptr<CIS::Distribution> incompBoards = cis1.computeDistributionGPU();
    std::uint64_t mostProb = cis1.computeMostProbableBoard(*incompBoards);
    EXPECT_EQ(mostProb,0);
}
};
