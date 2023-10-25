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
    ChessInformationSet cis;
    int a=1;


public:
    RBCAgent(NeuralNetAPI* netSingle,
              vector<unique_ptr<NeuralNetAPI>>& netBatches,
              SearchSettings* searchSettings,
              PlaySettings* playSettings,
              int iterations,
              bool splitNodes);
    //~RBCAgent();
    RBCAgent(const RBCAgent&) = delete;
    RBCAgent& operator=(RBCAgent const&) = delete;

    string get_name() const override;
    void evaluate_board_state() override;
};


#endif // RBCAGENT_H
