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
 * @file: rbcagent.h
 * Created on 10.2023
 * @author: meierh
 *
 * This RBCAgent is based on MCTSAgentBatch and has observation properties. 
 */

#ifndef RBCAGENT_H
#define RBCAGENT_H

#include <gtest/gtest.h>
#include "mctsagentbatch.h"
#include "chessinformationset.h"
#include <stdexcept>
#include <iostream>
#include <random>

namespace crazyara {

class RBCAgent : public MCTSAgent
{
    using CIS = ChessInformationSet;
    
public:
    enum PieceColor {white=0,black=1,empty=-1};
    static std::string combinedAgentsFEN
    (
        const RBCAgent& white,
        const RBCAgent& black,
        const PieceColor nextTurn,
        const unsigned int nextCompleteTurn
    );
    
private:   
    enum Player {Self, Opponent};
    enum MovePhase : std::uint8_t {Sense=0,Move=1};
    
    static open_spiel::chess::Color AgentColor_to_OpenSpielColor(const PieceColor agent_pC);
    static PieceColor OpenSpielColor_to_RBCColor(const open_spiel::chess::Color os_pC);
    
    static open_spiel::rbc::MovePhase AgentPhase_to_OpenSpielPhase(const MovePhase agent_pC);
    static MovePhase OpenSpielPhase_to_RBCPhase(const open_spiel::rbc::MovePhase os_pC);
    
    class FullChessInfo
    {
        public:
            FullChessInfo(){};
            FullChessInfo(std::string fen);
            
            CIS::OnePlayerChessInfo white;
            CIS::OnePlayerChessInfo black;
            
            MovePhase currentPhase;
            bool lastMoveCapturedPiece;
            PieceColor nextTurn;
            bool lastMoveIllegal;
            unsigned int nextCompleteTurn;
            
            static std::string getFEN
            (
                const CIS::OnePlayerChessInfo& white,
                const CIS::OnePlayerChessInfo& black,
                const PieceColor nextTurn,
                const unsigned int nextCompleteTurn
            );
            
            static void getAllFEN_GPU
            (
                CIS::OnePlayerChessInfo& self,
                const PieceColor selfColor,
                std::unique_ptr<ChessInformationSet>& cis,
                const PieceColor nextTurn,
                const unsigned int nextCompleteTurn,
                std::vector<std::string>& allFEN
            );
            
            std::string getFEN() const
            {
                return getFEN(this->white,this->black,nextTurn,nextCompleteTurn);
            };
            
            static void check
            (
                const CIS::OnePlayerChessInfo& white,
                const CIS::OnePlayerChessInfo& black,
                std::string lastAction=""
            );
            
            static std::array<std::pair<CIS::PieceType,PieceColor>,64> decodeFENFigurePlacement(std::string);
            
            static void splitFEN(std::string fen, std::vector<std::string>& fenParts);
    };

    CIS::OnePlayerChessInfo playerPiecesTracker;
    PieceColor selfColor;
    PieceColor opponentColor;
    unsigned int currentTurn;
    
    EvalInfo* evalInfo = nullptr;
    StateObj* rbcState;
    StateObj* chessState;
    
    void splitObsFEN(std::string obsFen,std::vector<std::string>& obsFenParts) const;
    void completeMoveData(open_spiel::chess::Move& move,CIS::OnePlayerChessInfo& opponentInfo) const;
    
protected:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<unsigned short> randomScanDist;
    std::uniform_int_distribution<std::uint64_t> randomHypotheseSelect;
    std::unique_ptr<ChessInformationSet> cis;
    
    std::pair<std::array<std::uint8_t,9>,std::array<CIS::Square,9>> getSensingBoardIndexes(CIS::Square sq);
    
public:
    RBCAgent
    (
        NeuralNetAPI* netSingle,
        vector<unique_ptr<NeuralNetAPI>>& netBatches,
        SearchSettings* searchSettings,
        PlaySettings* playSettings,
        std::string fen,
        PieceColor selfColor
    );
    //~RBCAgent();
    RBCAgent(const RBCAgent&) = delete;
    RBCAgent& operator=(RBCAgent const&) = delete;
    
    PieceColor getColor() {return selfColor;};
    
    /**
     * @brief set_search_settings 
     * Implements the InformationSet reduction and the observation in rbc
     * Sets all relevant parameters for the next search
     * @param pos Board position to evaluate
     * @param limits Pointer to the search limit
     * @param evalInfo Returns the evaluation information
     */
    void set_search_settings
    (
        StateObj *state,
        SearchLimits* searchLimits,
        EvalInfo* evalInfo
    );
    
    /**
     * @brief perform_action Selects an action based on the evaluation result
     */
    void perform_action();
    
    /**
     * @brief
     * @param piecesSelf
     * @param piecesOpponent
     * @param selfColor
     * @return
     */
    std::unique_ptr<std::vector<std::pair<ChessInformationSet::OnePlayerChessInfo,double>>> generateHypotheses
    (
        ChessInformationSet::OnePlayerChessInfo& piecesOpponent,
        ChessInformationSet::OnePlayerChessInfo& piecesSelf,
        const PieceColor selfColor,
        const std::vector<ChessInformationSet::BoardClause> conditions = {}
    ) const;

private:
    /**
     * @brief
     * @param piecesSelf
     * @return
     */
    std::unique_ptr<std::vector<std::pair<ChessInformationSet::OnePlayerChessInfo,double>>> generateHypotheses
    (
        ChessInformationSet::OnePlayerChessInfo& piecesOpponent,
        const std::vector<ChessInformationSet::BoardClause> conditions = {}
    );
    
    /**
     * @brief
     * @param pos
     * @param side
     * @return
     */
    std::unique_ptr<FullChessInfo> decodeObservation
    (
        StateObj *pos,
        PieceColor observerColor,
        PieceColor observationTargetColor
    ) const;
    
    std::unique_ptr<std::vector<float>> encodeStatePlane
    (
        const std::unique_ptr<FullChessInfo> fullState,
        const PieceColor nextTurn,
        const unsigned int nextCompleteTurn
    ) const;
    
    /**
     * @brief
     * @param pos
     */
    void handleOpponentMoveInfo
    (
        StateObj *pos
    );
    
    /**
     * @brief
     * @param pos
     */
    void handleSelfMoveInfo
    (
        StateObj *pos
    );
       
    /**
     * @brief
     * @param pos
     */
    ChessInformationSet::Square applyScanAction
    (
        StateObj *pos
    );
    /**
     * @brief
     * @param pos
     */
    ChessInformationSet::Square selectScanAction
    (
        StateObj *pos
    );    
    /**
     * @brief
     * @param pos
     * @param scanCenter
     */
    void handleScanInfo
    (
        StateObj *pos,
        ChessInformationSet::Square scanCenter
    );
    
    /**
     * @brief
     */
    std::unique_ptr<FullChessInfo> selectHypothese();
    
    /**
     * @brief
     */
    StateObj* setupMoveSearchState();
    
    /**
     * @brief
     */
    //void stepForwardHypotheses();
    
    FRIEND_TEST(rbcagentfullchessinfo_test, FEN_test);
    FRIEND_TEST(rbcagentfullchessinfo_test, FENReconstruction_test);
    FRIEND_TEST(rbcagentfullchessinfo_test, FENReconstructionGPU_test);
    FRIEND_TEST(rbcagentfullchessinfo_test, FENSplitting_test);
    FRIEND_TEST(rbcagentfullchessinfo_test, DecodeFENFigurePlacement_test);
    FRIEND_TEST(rbcagentfullchessinfo_test, Observation_test);
    FRIEND_TEST(chessinformationset_test, boardClause_test);
    FRIEND_TEST(chessinformationset_test, getIncompatibleGPU_test);
    FRIEND_TEST(chessinformationset_test, getDistributionGPU_test);
};
}

#endif // RBCAGENT_H
