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
    //Full info
    std::string fenIni = RBCAgent::FullChessInfo::getFEN(whiteSet,blackSet,RBCAgent::PieceColor::white,1);
    EXPECT_EQ(fenIni,"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    whiteSet.pawns[4] = SQ(COL::E,ROW::four);
    whiteSet.no_progress_count = 0;
    blackSet.no_progress_count++;
    whiteSet.en_passant.push_back(SQ(COL::E,ROW::three));
    fenIni = RBCAgent::FullChessInfo::getFEN(whiteSet,blackSet,RBCAgent::PieceColor::black,1);
    EXPECT_EQ(fenIni,"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    
    blackSet.pawns[2] = SQ(COL::C,ROW::five);
    whiteSet.no_progress_count++;
    blackSet.no_progress_count = 0;
    whiteSet.en_passant.clear();
    blackSet.en_passant.push_back(SQ(COL::C,ROW::six));
    fenIni = RBCAgent::FullChessInfo::getFEN(whiteSet,blackSet,RBCAgent::PieceColor::white,2);
    EXPECT_EQ(fenIni,"rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
    
    whiteSet.knights[1] = SQ(COL::F,ROW::three);
    whiteSet.no_progress_count++;
    blackSet.no_progress_count++;
    blackSet.en_passant.clear();
    whiteSet.no_progress_count++;
    fenIni = RBCAgent::FullChessInfo::getFEN(whiteSet,blackSet,RBCAgent::PieceColor::black,2);
    EXPECT_EQ(fenIni,"rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");    
}
};
