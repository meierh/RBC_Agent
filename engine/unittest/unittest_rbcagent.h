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

TEST(rbcagentfullchessinfo_test, Observation_test)
{
    using CIS = crazyara::ChessInformationSet;
    using COL = CIS::ChessColumn;
    using ROW = CIS::ChessRow;
    using SQ = CIS::Square;
    
    std::string rbcModelsDir = "/home/helge/Uni/Semester_21/Bachelorthesis/RBC_Agent/model/params";
    SearchSettings sSet;
    PlaySettings pSet;
    auto netSingle = std::make_unique<TensorrtAPI>(int(0), 1, rbcModelsDir, "float32");
    vector<unique_ptr<NeuralNetAPI>> netBatches;
    for(int i=0;i<sSet.threads;i++)
        netBatches.push_back(make_unique<TensorrtAPI>(int(0), 1, rbcModelsDir, "float32"));
    std::string initialFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    auto whitePlayer = std::make_unique<RBCAgent>(netSingle.get(),netBatches, &sSet, &pSet, initialFen, RBCAgent::PieceColor::white);
    auto blackPlayer = std::make_unique<RBCAgent>(netSingle.get(),netBatches, &sSet, &pSet, initialFen, RBCAgent::PieceColor::black);
    OpenSpielState rbcState(open_spiel::gametype::SupportedOpenSpielVariants::RBC);

    
    std::string fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    rbcState.set(fen1,false,open_spiel::gametype::SupportedOpenSpielVariants::RBC);
    std::unique_ptr<RBCAgent::FullChessInfo> fullObs1;
    fullObs1 = whitePlayer->decodeObservation(&rbcState,RBCAgent::PieceColor::white,RBCAgent::PieceColor::black);
    
    CIS::OnePlayerChessInfo& whiteObs1 = fullObs1->white;
    CIS::OnePlayerChessInfo whiteRef;
    whiteRef.pawns = {SQ(0,1),SQ(1,1),SQ(2,1),SQ(3,1),SQ(4,1),SQ(5,1),SQ(6,1),SQ(7,1)};
    whiteRef.knights = {SQ(1,0),SQ(6,0)};
    whiteRef.bishops = {SQ(2,0),SQ(5,0)};
    whiteRef.rooks = {SQ(0,0),SQ(7,0)};
    whiteRef.queens = {SQ(3,0)};
    whiteRef.kings = {SQ(4,0)};
    whiteRef.kingside = true;
    whiteRef.queenside = true;
    whiteRef.en_passant = {};
    whiteRef.no_progress_count = 0;
    EXPECT_EQ(whiteObs1,whiteRef);


    
    /*
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
    */
}
};
