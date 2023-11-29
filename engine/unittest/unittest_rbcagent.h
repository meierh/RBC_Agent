#include <iostream>
#include "rbcagent.h"

namespace crazyara {

TEST(rbcagentfullchessinfo_test, FEN_test)
{
    using CIS = crazyara::ChessInformationSet;
    using COL = CIS::ChessColumn;
    using ROW = CIS::ChessRow;
    using SQ = CIS::Square;
    
    //black
    CIS::OnePlayerChessInfo blackSet;
    blackSet.pawns = {  SQ(COL::A,ROW::seven),SQ(COL::B,ROW::seven),SQ(COL::C,ROW::seven),SQ(COL::D,ROW::seven),
                        SQ(COL::E,ROW::seven),SQ(COL::F,ROW::seven),SQ(COL::G,ROW::seven),SQ(COL::H,ROW::seven)};
    blackSet.knights={  SQ(COL::B,ROW::eight),SQ(COL::G,ROW::eight)};
    blackSet.bishops={  SQ(COL::C,ROW::eight),SQ(COL::F,ROW::eight)};
    blackSet.rooks = {  SQ(COL::A,ROW::eight),SQ(COL::H,ROW::eight)};
    blackSet.queens= {  SQ(COL::D,ROW::eight)};
    blackSet.kings = {  SQ(COL::E,ROW::eight)};
    
    //white
    CIS::OnePlayerChessInfo whiteSet;
    whiteSet.pawns = {  SQ(COL::A,ROW::two),SQ(COL::B,ROW::two),SQ(COL::C,ROW::two),SQ(COL::D,ROW::two),
                        SQ(COL::E,ROW::two),SQ(COL::F,ROW::two),SQ(COL::G,ROW::two),SQ(COL::H,ROW::two)};
    whiteSet.knights={  SQ(COL::B,ROW::one),SQ(COL::G,ROW::one)};
    whiteSet.bishops={  SQ(COL::C,ROW::one),SQ(COL::F,ROW::one)};
    whiteSet.rooks = {  SQ(COL::A,ROW::one),SQ(COL::H,ROW::one)};
    whiteSet.queens= {  SQ(COL::D,ROW::one)};
    whiteSet.kings = {  SQ(COL::E,ROW::one)};
    
    //RBCAgent::PieceColor color = RBCAgent::PieceColor::white;
    
    //Full info
    //RBCAgent::FullChessInfo::getFEN(whiteSet,blackSet,RBCAgent::PieceColor::white,1);// = {whiteInfo,blackInfo};
}
};
