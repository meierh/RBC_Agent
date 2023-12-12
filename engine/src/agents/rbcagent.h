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
    
private:   
    enum Player {Self, Opponent};
    enum PieceColor {white=0,black=1,empty=-1};
    enum MovePhase {sense=0,move=1};
    static open_spiel::chess::Color AgentColor_to_OpenSpielColor(const PieceColor agent_pC);
    static PieceColor OpenSpielColor_to_RBCColor(const open_spiel::chess::Color os_pC);
    
    class FullChessInfo
    {
        public:
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
            
            std::string getFEN() const
            {
                return getFEN(this->white,this->black,nextTurn,nextCompleteTurn);
            };
    };

    CIS::OnePlayerChessInfo playerPiecesTracker;
    PieceColor selfColor;
    unsigned int currentTurn;
    
    EvalInfo* evalInfo = nullptr;
    StateObj* chessOpenSpiel;

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
        PlaySettings* playSettings
    );
    //~RBCAgent();
    RBCAgent(const RBCAgent&) = delete;
    RBCAgent& operator=(RBCAgent const&) = delete;
    
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
        const PieceColor selfColor
    ) const;

private:
    /**
     * @brief
     * @param piecesSelf
     * @return
     */
    std::unique_ptr<std::vector<std::pair<ChessInformationSet::OnePlayerChessInfo,double>>> generateHypotheses
    (
        ChessInformationSet::OnePlayerChessInfo& piecesOpponent
    );
    
    /**
     * @brief
     * @param pos
     * @param side
     * @return
     */
    std::unique_ptr<FullChessInfo> decodeObservation
    (
        StateObj *pos
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
    void stepForwardHypotheses();
    
    FRIEND_TEST(rbcagentfullchessinfo_test, FEN_test);
};
}

#endif // RBCAGENT_H
