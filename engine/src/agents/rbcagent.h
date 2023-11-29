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

/*
#include "../evalinfo.h"
#include "../node.h"
#include "../stateobj.h"
#include "../nn/neuralnetapi.h"
#include "config/searchsettings.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#include "../searchthread.h"
#include "../manager/timemanager.h"
#include "../manager/threadmanager.h"
#include "util/gcthread.h"
*/

namespace crazyara {

class RBCAgent : public MCTSAgentBatch
{
    using CIS = ChessInformationSet;
    
private:   
    enum Player {Self, Opponent};
    enum PieceColor {white=0,black=1,empty=-1};
    static open_spiel::chess::Color AgentColor_to_OpenSpielColor(const PieceColor agent_pC);
    static PieceColor OpenSpielColor_to_RBCColor(const open_spiel::chess::Color os_pC);
    
    class FullChessInfo
    {
        public:
            CIS::OnePlayerChessInfo white;
            CIS::OnePlayerChessInfo black;
            
            static std::string getFEN
            (
                const CIS::OnePlayerChessInfo& white,
                const CIS::OnePlayerChessInfo& black,
                const PieceColor nextTurn,
                const unsigned int nextCompleteTurn
            );
            
            std::string getFEN
            (
                const PieceColor nextTurn,
                const unsigned int nextCompleteTurn
            ) const
            {
                return getFEN(this->white,this->black,nextTurn,nextCompleteTurn);
            };
    };

    ChessInformationSet cis;
    CIS::OnePlayerChessInfo playerPiecesTracker;
    PieceColor selfColor;
    unsigned int currentTurn;

public:
    RBCAgent
    (
        NeuralNetAPI* netSingle,
        vector<unique_ptr<NeuralNetAPI>>& netBatches,
        SearchSettings* searchSettings,
        PlaySettings* playSettings,
        int iterations,
        bool splitNodes
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
     * @brief
     * @param piecesSelf
     * @param piecesOpponent
     * @param selfColor
     * @return
     */
    std::unique_ptr<std::vector<ChessInformationSet::OnePlayerChessInfo>> generateHypotheses
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
    std::unique_ptr<std::vector<ChessInformationSet::OnePlayerChessInfo>> generateHypotheses
    (
        ChessInformationSet::OnePlayerChessInfo& piecesOpponent
    );
    
    /**
     * @brief
     * @param pos
     * @param side
     * @return
     */
    std::unique_ptr<FullChessInfo> getDecodedStatePlane
    (
        StateObj *pos
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
    ChessInformationSet::Square applyScanAction
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
    void handleMoveInfo();
    
    FRIEND_TEST(rbcagentfullchessinfo_test, FEN_test);
};
}

#endif // RBCAGENT_H
