#include <iostream>
#include "nn/tensorrtapi.h"
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
    whiteSet.en_passant_valid = true;
    whiteSet.en_passant = SQ(COL::E,ROW::three);
    fenIni = RBCAgent::FullChessInfo::getFEN(whiteSet,blackSet,RBCAgent::PieceColor::black,1);
    EXPECT_EQ(fenIni,"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    
    blackSet.pawns[2] = SQ(COL::C,ROW::five);
    whiteSet.no_progress_count++;
    blackSet.no_progress_count = 0;
    blackSet.en_passant_valid = true;
    blackSet.en_passant = SQ(COL::C,ROW::six);
    fenIni = RBCAgent::FullChessInfo::getFEN(whiteSet,blackSet,RBCAgent::PieceColor::white,2);
    EXPECT_EQ(fenIni,"rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
    
    whiteSet.knights[1] = SQ(COL::F,ROW::three);
    whiteSet.no_progress_count++;
    blackSet.no_progress_count++;
    whiteSet.en_passant_valid = false;
    whiteSet.no_progress_count++;
    fenIni = RBCAgent::FullChessInfo::getFEN(whiteSet,blackSet,RBCAgent::PieceColor::black,2);
    EXPECT_EQ(fenIni,"rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");    
}

TEST(rbcagentfullchessinfo_test, FENReconstruction_test)
{
    using CIS = crazyara::ChessInformationSet;
    using COL = CIS::ChessColumn;
    using ROW = CIS::ChessRow;
    using SQ = CIS::Square;
    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    RBCAgent::FullChessInfo fci1(fen1);
    EXPECT_EQ(fci1.getFEN(),fen1);
    
    std::string fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    RBCAgent::FullChessInfo fci2(fen2);
    EXPECT_EQ(fci2.getFEN(),fen2);
    
    std::string fen3 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2";
    RBCAgent::FullChessInfo fci3(fen3);
    EXPECT_EQ(fci3.getFEN(),fen3);
    
    std::string fen4 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
    RBCAgent::FullChessInfo fci4(fen4);
    EXPECT_EQ(fci4.getFEN(),fen4);
    
    std::string fen5 = "6r1/6pp/7r/1B5K/1P3k2/N7/3R4/8 w - - 30 79";
    RBCAgent::FullChessInfo fci5(fen5);
    EXPECT_EQ(fci5.getFEN(),fen5);
    
    std::string fen6 = "2k5/5P1K/3P2p1/3P2NP/p3PBp1/5B2/Q4n2/6r1 w - - 0 1";
    RBCAgent::FullChessInfo fci6(fen6);
    EXPECT_EQ(fci6.getFEN(),fen6);

    std::string fen7 = "5r2/p1Q3pb/5p1p/4PP2/6p1/8/pBR1Nk2/3K4 w - - 0 1";
    RBCAgent::FullChessInfo fci7(fen7);
    EXPECT_EQ(fci7.getFEN(),fen7);
    
    std::string fen8 = "8/5pPn/kP2b3/P1PP1N2/5p1K/r1p5/P6p/8 w - - 0 1";
    RBCAgent::FullChessInfo fci8(fen8);
    EXPECT_EQ(fci8.getFEN(),fen8);
    
    std::string fen9 = "3b2B1/4p3/k2p3K/2P1nq2/4bP2/6P1/2p1r1P1/7Q w - - 0 1";
    RBCAgent::FullChessInfo fci9(fen9);
    EXPECT_EQ(fci9.getFEN(),fen9);
    
    std::string fen10 = "2r5/2P3p1/2Q1p3/1P1p3B/2P4K/3p4/p1NP1k2/q7 w - - 0 1";
    RBCAgent::FullChessInfo fci10(fen10);
    EXPECT_EQ(fci10.getFEN(),fen10);
    
    std::string fen11 = "7N/8/8/8/5k2/8/1PK5/8 w - - 0 1";
    RBCAgent::FullChessInfo fci11(fen11);
    EXPECT_EQ(fci11.getFEN(),fen11);
    
    std::string fen12 = "8/8/8/6k1/8/K7/8/8 w - - 0 1";
    RBCAgent::FullChessInfo fci12(fen12);
    EXPECT_EQ(fci12.getFEN(),fen12);
    
    std::string fen13 = "2rn1b2/2pPNQ1P/K3Bpb1/2PPPpp1/np1kN1RP/p4p2/P1Pqp2R/2B4r w - - 0 1";
    RBCAgent::FullChessInfo fci13(fen13);
    EXPECT_EQ(fci13.getFEN(),fen13);
}

TEST(rbcagentfullchessinfo_test, FENReconstructionGPU_test)
{
    using CIS = crazyara::ChessInformationSet;
    using COL = CIS::ChessColumn;
    using ROW = CIS::ChessRow;
    using SQ = CIS::Square;
    
    std::vector<std::string> fens = 
    {
        "rnbqkbnr/pppppppp/8/8/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kkq - 0 1",
        "rnbqkbnr/pp1pp1pp/2p5/5p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kkq f6 0 1",
        "rnb1kbnr/pp1pp1pp/1qp5/5p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kkq - 0 1",
        "rnb1kbnr/pp2p1pp/1qpp4/5p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kkq - 0 1",
        "rnb1kbnr/1p2p1pp/pqpp4/5p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kkq - 0 1",
        "rnb1kbnr/1p2p1pp/pqp5/3p1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kkq - 0 1",
        "rnb1kb1r/1p2p1pp/pqp2n2/3p1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kkq - 0 1",
        "1nb1kb1r/rp2p1pp/pqp2n2/3p1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kk - 0 1",
        "1n2kb1r/rp2p1pp/pqp1bn2/3p1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kk - 0 1",
        "1n2kb1r/rp4pp/pqppbn2/3p1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kk - 0 1",
        "1n2kb1r/rp4pp/pq1pbn2/2pp1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kk - 0 1",
        "1n2kb1r/rp5p/1q1pbnp1/p1pp1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kk - 0 1",
        "1n2k2r/rp5p/1q1pbnpb/p1pp1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w Kk - 0 1",
        "1n3rk1/rp5p/1q1pbnpb/p1pp1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w K - 0 1",
        "1n3rk1/rp1b3p/1q1p1npb/p1pp1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w K - 0 1",
        "1n3rk1/rp1b3p/3p1npb/pqpp1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w K - 0 1",
        "1n3rk1/rp5p/2bp1npb/pqpp1p2/1B2P3/NP1Q1NPP/P1P1P1P1/1R2KB1R w K - 0 1",
        "1n3rk1/rp5p/2bp1npb/pqpp4/1B2P1p1/NP1Q1NPP/P1P1P1P1/1R2KB1R w K - 0 1",
        "1n3rk1/rp5p/2bp1np1/pqpp4/1B2Pbp1/NP1Q1NPP/P1P1P1P1/1R2KB1R w K - 0 1"
    };
    
    auto cisUPtr = std::make_unique<CIS>();
    CIS& cis = *cisUPtr;
    for(std::string fen : fens)
    {
        RBCAgent::FullChessInfo fci(fen);
        cis.add(fci.black,1);
    }
    
    RBCAgent::FullChessInfo fciSelf(fens[0]);
    
    std::vector<std::string> allFEN;
    RBCAgent::FullChessInfo::getAllFEN_GPU
    (
        fciSelf.white,
        RBCAgent::PieceColor::white,
        cisUPtr,
        RBCAgent::PieceColor::white,
        1,
        allFEN
    );
    
    for(int i=0; i<allFEN.size(); i++)
    {
        EXPECT_EQ(allFEN[i],fens[i]);
    }
}

TEST(rbcagentfullchessinfo_test, FENSplitting_test)
{
    using CIS = crazyara::ChessInformationSet;
    using COL = CIS::ChessColumn;
    using ROW = CIS::ChessRow;
    using SQ = CIS::Square;
    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    std::vector<std::string> fen1Parts = {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR","w","KQkq","-","0","1"};
    std::vector<std::string> fen1Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen1,fen1Splitted);
    EXPECT_EQ(fen1Splitted,fen1Parts);
    
    std::string fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    std::vector<std::string> fen2Parts = {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR","b","KQkq","e3","0","1"};
    std::vector<std::string> fen2Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen2,fen2Splitted);
    EXPECT_EQ(fen2Splitted,fen2Parts);
    
    std::string fen3 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2";
    std::vector<std::string> fen3Parts = {"rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR","w","KQkq","c6","0","2"};
    std::vector<std::string> fen3Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen3,fen3Splitted);
    EXPECT_EQ(fen3Splitted,fen3Parts);
    
    std::string fen4 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
    std::vector<std::string> fen4Parts = {"rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R","b","KQkq","-","1","2"};
    std::vector<std::string> fen4Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen4,fen4Splitted);
    EXPECT_EQ(fen4Splitted,fen4Parts);
    
    std::string fen5 = "6r1/6pp/7r/1B5K/1P3k2/N7/3R4/8 w - - 30 79";
    std::vector<std::string> fen5Parts = {"6r1/6pp/7r/1B5K/1P3k2/N7/3R4/8","w","-","-","30","79"};
    std::vector<std::string> fen5Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen5,fen5Splitted);
    EXPECT_EQ(fen5Splitted,fen5Parts);
    
    std::string fen6 = "2k5/5P1K/3P2p1/3P2NP/p3PBp1/5B2/Q4n2/6r1 w - - 0 1";
    std::vector<std::string> fen6Parts = {"2k5/5P1K/3P2p1/3P2NP/p3PBp1/5B2/Q4n2/6r1","w","-","-","0","1"};
    std::vector<std::string> fen6Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen6,fen6Splitted);
    EXPECT_EQ(fen6Splitted,fen6Parts);

    std::string fen7 = "5r2/p1Q3pb/5p1p/4PP2/6p1/8/pBR1Nk2/3K4 w - - 0 1";
    std::vector<std::string> fen7Parts = {"5r2/p1Q3pb/5p1p/4PP2/6p1/8/pBR1Nk2/3K4","w","-","-","0","1"};
    std::vector<std::string> fen7Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen7,fen7Splitted);
    EXPECT_EQ(fen7Splitted,fen7Parts);

    std::string fen8 = "8/5pPn/kP2b3/P1PP1N2/5p1K/r1p5/P6p/8 w - - 0 1";
    std::vector<std::string> fen8Parts = {"8/5pPn/kP2b3/P1PP1N2/5p1K/r1p5/P6p/8","w","-","-","0","1"};
    std::vector<std::string> fen8Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen8,fen8Splitted);
    EXPECT_EQ(fen8Splitted,fen8Parts);
    
    std::string fen9 = "3b2B1/4p3/k2p3K/2P1nq2/4bP2/6P1/2p1r1P1/7Q w - - 0 1";
    std::vector<std::string> fen9Parts = {"3b2B1/4p3/k2p3K/2P1nq2/4bP2/6P1/2p1r1P1/7Q","w","-","-","0","1"};
    std::vector<std::string> fen9Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen9,fen9Splitted);
    EXPECT_EQ(fen9Splitted,fen9Parts);
    
    std::string fen10 = "2r5/2P3p1/2Q1p3/1P1p3B/2P4K/3p4/p1NP1k2/q7 w - - 0 1";
    std::vector<std::string> fen10Parts = {"2r5/2P3p1/2Q1p3/1P1p3B/2P4K/3p4/p1NP1k2/q7","w","-","-","0","1"};
    std::vector<std::string> fen10Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen10,fen10Splitted);
    EXPECT_EQ(fen10Splitted,fen10Parts);
    
    std::string fen11 = "7N/8/8/8/5k2/8/1PK5/8 w - - 0 1";
    std::vector<std::string> fen11Parts = {"7N/8/8/8/5k2/8/1PK5/8","w","-","-","0","1"};
    std::vector<std::string> fen11Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen11,fen11Splitted);
    EXPECT_EQ(fen11Splitted,fen11Parts);
    
    std::string fen12 = "8/8/8/6k1/8/K7/8/8 w - - 0 1";
    std::vector<std::string> fen12Parts = {"8/8/8/6k1/8/K7/8/8","w","-","-","0","1"};
    std::vector<std::string> fen12Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen12,fen12Splitted);
    EXPECT_EQ(fen12Splitted,fen12Parts);
    
    std::string fen13 = "2rn1b2/2pPNQ1P/K3Bpb1/2PPPpp1/np1kN1RP/p4p2/P1Pqp2R/2B4r w - - 0 1";
    std::vector<std::string> fen13Parts = {"2rn1b2/2pPNQ1P/K3Bpb1/2PPPpp1/np1kN1RP/p4p2/P1Pqp2R/2B4r","w","-","-","0","1"};
    std::vector<std::string> fen13Splitted;
    RBCAgent::FullChessInfo::splitFEN(fen13,fen13Splitted);
    EXPECT_EQ(fen13Splitted,fen13Parts);
}

TEST(rbcagentfullchessinfo_test, DecodeFENFigurePlacement_test)
{
    using CIS = crazyara::ChessInformationSet;
    using COL = CIS::ChessColumn;
    using ROW = CIS::ChessRow;
    using SQ = CIS::Square;
    
    std::pair<CIS::PieceType,RBCAgent::PieceColor> None = {CIS::PieceType::unknown,RBCAgent::PieceColor::empty};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> PaW = {CIS::PieceType::pawn,RBCAgent::PieceColor::white};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> RoW = {CIS::PieceType::rook,RBCAgent::PieceColor::white};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> KnW = {CIS::PieceType::knight,RBCAgent::PieceColor::white};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> BiW = {CIS::PieceType::bishop,RBCAgent::PieceColor::white};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> QuW = {CIS::PieceType::queen,RBCAgent::PieceColor::white};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> KiW = {CIS::PieceType::king,RBCAgent::PieceColor::white};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> PaB = {CIS::PieceType::pawn,RBCAgent::PieceColor::black};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> RoB = {CIS::PieceType::rook,RBCAgent::PieceColor::black};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> KnB = {CIS::PieceType::knight,RBCAgent::PieceColor::black};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> BiB = {CIS::PieceType::bishop,RBCAgent::PieceColor::black};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> QuB = {CIS::PieceType::queen,RBCAgent::PieceColor::black};
    std::pair<CIS::PieceType,RBCAgent::PieceColor> KiB = {CIS::PieceType::king,RBCAgent::PieceColor::black};

    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,64> figureBoardRef;
    std::fill(figureBoardRef.begin(),figureBoardRef.end(),None);
    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,8> row0 = {RoW,KnW,BiW,QuW,KiW,BiW,KnW,RoW};
    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,8> row7 = {RoB,KnB,BiB,QuB,KiB,BiB,KnB,RoB};
    for(int col=0;col<8;col++)
    {
        figureBoardRef[CIS::squareToBoardIndex(CIS::Square(col,0))] = row0[col];
        figureBoardRef[CIS::squareToBoardIndex(CIS::Square(col,1))] = PaW;
        figureBoardRef[CIS::squareToBoardIndex(CIS::Square(col,6))] = PaB;
        figureBoardRef[CIS::squareToBoardIndex(CIS::Square(col,7))] = row7[col];
    }
    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,64> figureBoard1 = RBCAgent::FullChessInfo::decodeFENFigurePlacement(fen1);
    EXPECT_EQ(figureBoard1,figureBoardRef);
    
    std::string fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR";
    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,64> figureBoard2 = RBCAgent::FullChessInfo::decodeFENFigurePlacement(fen2);
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(4,1))] = None;
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(4,3))] = PaW;
    EXPECT_EQ(figureBoard2,figureBoardRef);

    std::string fen3 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR";
    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,64> figureBoard3 = RBCAgent::FullChessInfo::decodeFENFigurePlacement(fen3);
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(2,6))] = None;
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(2,4))] = PaB;
    EXPECT_EQ(figureBoard3,figureBoardRef);

    std::string fen4 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R";
    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,64> figureBoard4 = RBCAgent::FullChessInfo::decodeFENFigurePlacement(fen4);
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(6,0))] = None;
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(5,2))] = KnW;
    EXPECT_EQ(figureBoard4,figureBoardRef);

    std::string fen5 = "7N/8/8/8/5k2/8/1PK5/8";
    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,64> figureBoard5 = RBCAgent::FullChessInfo::decodeFENFigurePlacement(fen5);
    std::fill(figureBoardRef.begin(),figureBoardRef.end(),None);
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(7,7))] = KnW;
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(5,3))] = KiB;
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(1,1))] = PaW;
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(2,1))] = KiW;
    EXPECT_EQ(figureBoard5,figureBoardRef);
    
    std::string fen6 = "8/8/8/6k1/8/K7/8/8";
    std::array<std::pair<CIS::PieceType,RBCAgent::PieceColor>,64> figureBoard6 = RBCAgent::FullChessInfo::decodeFENFigurePlacement(fen6);
    std::fill(figureBoardRef.begin(),figureBoardRef.end(),None);
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(0,2))] = KiW;
    figureBoardRef[CIS::squareToBoardIndex(CIS::Square(6,4))] = KiB;
    EXPECT_EQ(figureBoard6,figureBoardRef);

}
/*
TEST(rbcagentfullchessinfo_test, Observation_test)
{
    using CIS = crazyara::ChessInformationSet;
    using COL = CIS::ChessColumn;
    using ROW = CIS::ChessRow;
    using SQ = CIS::Square;
    
    std::string rbcModelsDir = "/home/helge/Uni/Semester_21/Bachelorthesis/RBC_Agent/model/params";
    SearchSettings sSet;
    PlaySettings pSet;
    std::cout<<"IniCreated NN"<<std::endl;
    auto netSingle = std::make_unique<TensorrtAPI>(int(0), 1, rbcModelsDir, "float32");
    std::cout<<"Created NN"<<std::endl;
    vector<unique_ptr<NeuralNetAPI>> netBatches;
    for(int i=0;i<sSet.threads;i++)
        netBatches.push_back(make_unique<TensorrtAPI>(int(0), 1, rbcModelsDir, "float32"));
    std::cout<<"Created NN Batch"<<std::endl;
    std::string initialFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    auto whitePlayer = std::make_unique<RBCAgent>(netSingle.get(),netBatches, &sSet, &pSet, initialFen, RBCAgent::PieceColor::white);
    std::cout<<"Created White"<<std::endl;
    auto blackPlayer = std::make_unique<RBCAgent>(netSingle.get(),netBatches, &sSet, &pSet, initialFen, RBCAgent::PieceColor::black);
    std::cout<<"Created Black"<<std::endl;
    OpenSpielState rbcState(open_spiel::gametype::SupportedOpenSpielVariants::RBC);

    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    rbcState.set(fen1,false,open_spiel::gametype::SupportedOpenSpielVariants::RBC);
    std::unique_ptr<RBCAgent::FullChessInfo> fullObsWhite1;
    fullObsWhite1 = whitePlayer->decodeObservation(&rbcState,RBCAgent::PieceColor::white,RBCAgent::PieceColor::black);
    
    CIS::OnePlayerChessInfo& whiteObs1 = fullObsWhite1->white;
    CIS::OnePlayerChessInfo whiteRef1;
    whiteRef1.pawns = {SQ(0,1),SQ(1,1),SQ(2,1),SQ(3,1),SQ(4,1),SQ(5,1),SQ(6,1),SQ(7,1)};
    whiteRef1.knights = {SQ(1,0),SQ(6,0)};
    whiteRef1.bishops = {SQ(2,0),SQ(5,0)};
    whiteRef1.rooks = {SQ(0,0),SQ(7,0)};
    whiteRef1.queens = {SQ(3,0)};
    whiteRef1.kings = {SQ(4,0)};
    whiteRef1.kingside = true;
    whiteRef1.queenside = true;
    whiteRef1.en_passant = {};
    whiteRef1.no_progress_count = 0;
    EXPECT_EQ(whiteObs1,whiteRef1);
    
    CIS::OnePlayerChessInfo& blackObs1 = fullObsWhite1->black;
    CIS::OnePlayerChessInfo blackRef1;
    blackRef1.pawns = {};
    blackRef1.knights = {};
    blackRef1.bishops = {};
    blackRef1.rooks = {};
    blackRef1.queens = {};
    blackRef1.kings = {};
    blackRef1.kingside = true;
    blackRef1.queenside = true;
    blackRef1.en_passant = {};
    blackRef1.no_progress_count = 0;
    EXPECT_EQ(blackObs1,blackRef1);

    rbcState.do_action(CIS::squareToBoardIndex(SQ(2,6)));
    fullObsWhite1 = whitePlayer->decodeObservation(&rbcState,RBCAgent::PieceColor::white,RBCAgent::PieceColor::black);
    
    CIS::OnePlayerChessInfo& blackObs2 = fullObsWhite1->black;
    CIS::OnePlayerChessInfo blackRef2;
    blackRef2.pawns = {SQ(1,6),SQ(2,6),SQ(3,6)};
    blackRef2.knights = {SQ(1,7)};
    blackRef2.bishops = {SQ(2,7)};
    blackRef2.rooks = {};
    blackRef2.queens = {SQ(3,7)};
    blackRef2.kings = {};
    blackRef2.kingside = true;
    blackRef2.queenside = true;
    blackRef2.en_passant = {};
    blackRef2.no_progress_count = 0;
    EXPECT_EQ(blackObs2,blackRef2);
    

    std::string fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    rbcState.set(fen2,false,open_spiel::gametype::SupportedOpenSpielVariants::RBC);
    std::cout<<"State created"<<std::endl;
    rbcState.do_action(CIS::squareToBoardIndex(SQ(4,2)));
    std::cout<<"Sensed"<<std::endl;
    std::unique_ptr<RBCAgent::FullChessInfo> fullObsBlack1;
    fullObsBlack1 = blackPlayer->decodeObservation(&rbcState,RBCAgent::PieceColor::black,RBCAgent::PieceColor::white);
    std::cout<<"Observed"<<std::endl;
    
    CIS::OnePlayerChessInfo& blackObs3 = fullObsBlack1->black;
    CIS::OnePlayerChessInfo blackRef3;
    blackRef3.pawns = {SQ(0,6),SQ(1,6),SQ(2,6),SQ(3,6),SQ(4,6),SQ(5,6),SQ(6,6),SQ(7,6)};
    blackRef3.knights = {SQ(1,7),SQ(6,7)};
    blackRef3.bishops = {SQ(2,7),SQ(5,7)};
    blackRef3.rooks = {SQ(0,7),SQ(7,7)};
    blackRef3.queens = {SQ(3,7)};
    blackRef3.kings = {SQ(4,7)};
    blackRef3.kingside = true;
    blackRef3.queenside = true;
    blackRef3.en_passant = {};
    blackRef3.no_progress_count = 0;
    EXPECT_EQ(blackObs3,blackRef3);
    
    CIS::OnePlayerChessInfo& whiteObs2 = fullObsBlack1->white;
    CIS::OnePlayerChessInfo whiteRef2;
    whiteRef2.pawns = {SQ(3,1),SQ(5,1),SQ(4,3)};
    whiteRef2.knights = {};
    whiteRef2.bishops = {};
    whiteRef2.rooks = {};
    whiteRef2.queens = {};
    whiteRef2.kings = {};
    whiteRef2.kingside = true;
    whiteRef2.queenside = true;
    whiteRef2.en_passant = {};
    whiteRef2.no_progress_count = 0;
    EXPECT_EQ(whiteObs2,whiteRef2);
    
    
    std::string fen3 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2";

    
    std::string fen4 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";

    
    std::string fen5 = "6r1/6pp/7r/1B5K/1P3k2/N7/3R4/8 w - - 30 79";

    
    std::string fen6 = "2k5/5P1K/3P2p1/3P2NP/p3PBp1/5B2/Q4n2/6r1 w - - 0 1";


    std::string fen7 = "5r2/p1Q3pb/5p1p/4PP2/6p1/8/pBR1Nk2/3K4 w - - 0 1";


    std::string fen8 = "8/5pPn/kP2b3/P1PP1N2/5p1K/r1p5/P6p/8 w - - 0 1";

    
    std::string fen9 = "3b2B1/4p3/k2p3K/2P1nq2/4bP2/6P1/2p1r1P1/7Q w - - 0 1";

    
    std::string fen10 = "2r5/2P3p1/2Q1p3/1P1p3B/2P4K/3p4/p1NP1k2/q7 w - - 0 1";

    
    std::string fen11 = "7N/8/8/8/5k2/8/1PK5/8 w - - 0 1";

    
    std::string fen12 = "8/8/8/6k1/8/K7/8/8 w - - 0 1";

    
    std::string fen13 = "2rn1b2/2pPNQ1P/K3Bpb1/2PPPpp1/np1kN1RP/p4p2/P1Pqp2R/2B4r w - - 0 1";
}
*/
};
