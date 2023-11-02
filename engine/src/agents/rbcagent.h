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

using namespace crazyara;

class RBCAgent : public MCTSAgentBatch
{
private:   
    class ChessPiecesObservation
    {
    public:
        std::vector<ChessInformationSet::Square> pawns;
        std::vector<ChessInformationSet::Square> knights;
        std::vector<ChessInformationSet::Square> bishops;
        std::vector<ChessInformationSet::Square> rooks;
        std::vector<ChessInformationSet::Square> queens;
        std::vector<ChessInformationSet::Square> kings;
    };
    enum Player {Self, Opponent};
    enum PieceColor {White,Black};

    ChessInformationSet cis;
    ChessInformationSet::ChessPiecesInformation playerPiecesTracker;
    PieceColor selfColor;

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
    std::unique_ptr<std::vector<ChessInformationSet::ChessPiecesInformation>> generateHypotheses
    (
        ChessInformationSet::ChessPiecesInformation& piecesOpponent,
        ChessInformationSet::ChessPiecesInformation& piecesSelf,
        const PieceColor selfColor
    ) const;

private:
    /**
     * @brief
     * @param piecesSelf
     * @return
     */
    std::unique_ptr<std::vector<ChessInformationSet::ChessPiecesInformation>> generateHypotheses
    (
        ChessInformationSet::ChessPiecesInformation& piecesOpponent
    );
    
    /**
     * @brief
     * @param pos
     * @param side
     * @return
     */
    std::unique_ptr<ChessPiecesObservation> getDecodedStatePlane
    (
        StateObj *pos,
        const Player side        
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
};

#endif // RBCAGENT_H
