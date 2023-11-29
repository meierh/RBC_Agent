#include <iostream>
#include "chessinformationset.h"

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
};
