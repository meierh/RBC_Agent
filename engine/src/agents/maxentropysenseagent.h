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
 * @file: maxentropysenseagent.h
 * Created on 12.2023
 * @author: meierh
 *
 * This RBCAgent is based on MCTSAgent and has observation properties. 
 */

#ifndef MAXENTROPYSENSEAGENT_H
#define MAXENTROPYSENSEAGENT_H

#include <gtest/gtest.h>
#include "rbcagent.h"
#include <stdexcept>
#include <iostream>

namespace crazyara {

class MaxEntropySenseAgent : public RBCAgent

{
public:
    MaxEntropySenseAgent
    (
        NeuralNetAPI* netSingle,
        vector<unique_ptr<NeuralNetAPI>>& netBatches,
        SearchSettings* searchSettings,
        PlaySettings* playSettings
    ):RBCAgent(netSingle,netBatches,searchSettings,playSettings){}

    MaxEntropySenseAgent(const MaxEntropySenseAgent&) = delete;
    MaxEntropySenseAgent& operator=(MaxEntropySenseAgent const&) = delete;

private:
    /**
     * @brief
     * @param pos
     */
    ChessInformationSet::Square selectScanAction
    (
        StateObj *pos
    );
};
}

#endif // RBCAGENT_H
