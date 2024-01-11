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
 * @file: boardstate.h
 * Created on 20.01.2021
 * @author: queensgambit
 */

#include "openspielstate.h"
#include "util/communication.h"
#include <functional>

OpenSpielState::OpenSpielState(uint8_t variant)
{
    if(variant>=4)
        throw std::invalid_argument("Game variant must be in [0,3]");
    open_spiel::gametype::SupportedOpenSpielVariants gameVariant;
    gameVariant = static_cast<open_spiel::gametype::SupportedOpenSpielVariants>(variant);
    currentVariant = gameVariant;
    spielGame = open_spiel::LoadGame(StateConstantsOpenSpiel::variant_to_string(currentVariant));
    spielState = spielGame->NewInitialState();
}

OpenSpielState::OpenSpielState(const OpenSpielState &openSpielState):
    currentVariant(openSpielState.currentVariant),
    spielGame(openSpielState.spielGame->shared_from_this()),
    spielState(openSpielState.spielState->Clone())
{
}

std::vector<Action> OpenSpielState::legal_actions(open_spiel::chess::Color actor) const
{
    int actorInt = open_spiel::chess::ColorToPlayer(actor);
    return spielState->LegalActions(actorInt);
}

std::vector<Action> OpenSpielState::legal_actions() const
{
    return spielState->LegalActions(spielState->CurrentPlayer());
}

inline void OpenSpielState::check_variant(int variant)
{
    if (variant != currentVariant) {
        currentVariant = open_spiel::gametype::SupportedOpenSpielVariants(variant);
        spielGame = open_spiel::LoadGame(StateConstantsOpenSpiel::variant_to_string(currentVariant));
    }
}

void OpenSpielState::set(const std::string &fenStr, bool isChess960, int variant)
{
    check_variant(variant);
    if (currentVariant == open_spiel::gametype::SupportedOpenSpielVariants::HEX)
    {
        info_string_important("NewInitialState from string is not implemented for HEX.");
        return;
    }
    spielState = spielGame->NewInitialState(fenStr);
}

void OpenSpielState::get_state_planes
(
    bool normalize,
    std::vector<float>& statePlanes,
    Version version,
    open_spiel::chess::Color observer,
    open_spiel::chess::Color observedTarget
) const
{
    int observerInt = open_spiel::chess::ColorToPlayer(observer);
    std::vector<float> v(spielGame->ObservationTensorSize());
    spielState->ObservationTensor(observerInt, absl::MakeSpan(v));
    statePlanes.resize(spielGame->ObservationTensorSize());
    std::memcpy(statePlanes.data(),v.data(),statePlanes.size()*sizeof(float));
}

void OpenSpielState::get_state_planes
(
    bool normalize,
    float *inputPlanes,
    Version version
) const
{
    //std::fill(inputPlanes, inputPlanes+StateConstantsOpenSpiel::NB_VALUES_TOTAL(), 0.0f);
    // TODO fix the double free error
    std::vector<float> v(spielGame->ObservationTensorSize());
    spielState->ObservationTensor(spielState->CurrentPlayer(), absl::MakeSpan(v));
    std::copy( v.begin(), v.end(), inputPlanes);

    /*    
    int offset=0;
    std::cout<<"Observation v Tensor"<<std::endl;
    for(int k=0; k<52; k++) // max 52
    {
        std::cout<<"inputPlanes offset:"<<offset<<std::endl;
        std::string observationStr;
        observationStr=observationStr+"    a  b  c  d  e  f  g  h \n";
        observationStr=observationStr+"    ---------------------- \n";
        for(int8_t rank=7;rank>=0;rank--)
        {
        observationStr=observationStr+std::to_string(rank+1)+" |";
        for(int8_t file=0;file<8;file++)
        {
            const size_t index = open_spiel::chess::SquareToIndex(open_spiel::chess::Square{file, rank}, 8);
            observationStr = observationStr + " " + std::to_string((bool)inputPlanes[offset+index]) + " ";
        }
        observationStr=observationStr+"\n";
        }
        observationStr=observationStr+"\n";
        std::cout<<observationStr<<std::endl;
        
        offset+=64;
    }
    */
}

std::string OpenSpielState::get_state_string
(
    open_spiel::chess::Color observer,
    open_spiel::chess::Color observedTarget
) const
{
    int observerInt = open_spiel::chess::ColorToPlayer(observer);
    //std::cout<<"Observation of RBC player:"<<spielState->CurrentPlayer()<<std::endl;
    std::string obsString = spielState->ObservationString(observerInt);
    //std::cout<<"obsString:"<<obsString<<std::endl;
    return obsString;
}

unsigned int OpenSpielState::steps_from_null() const
{
    return spielState->MoveNumber() / 2;  // note: MoveNumber != PlyCount
}

bool OpenSpielState::is_chess960() const
{
    return false;
}

std::string OpenSpielState::fen() const
{
    return spielState->ToString();
}

void OpenSpielState::do_action(Action action)
{
    if (currentVariant == open_spiel::gametype::SupportedOpenSpielVariants::HEX)
    {
        int X = action / 11; // currently easier to set board size fix; change it later
        int Y = action % 11;
        spielState->ApplyAction(Y*11+X);
        return;
    }
    /*
    string fen = spielState->ToString();
    string uciAction = spielState->ActionToString(spielState->CurrentPlayer(), action);
    std::cout << fen << "  Apply  " << action << "  " << uciAction << std::endl;
    */
    spielState->ApplyAction(action);
    //spielState->ApplyAction(001);   // dummy sense action
}

void OpenSpielState::undo_action(Action action)
{
    spielState->UndoAction(!spielState->CurrentPlayer(), action); // note: this formulation assumes a two player, non-simultaneaous game
}

void OpenSpielState::prepare_action()
{
    // pass
}

unsigned int OpenSpielState::number_repetitions() const
{
    // TODO
    return 0;
}

int OpenSpielState::side_to_move() const
{
    // spielState->CurrentPlayer()) may return negative values for terminal values.
    // This implementation assumes a two player game with ordered turns.
    // MoveNumber() assumes to be the number of plies and not chess moves.
    
    // TODO: MoveNumber no longer available
    return spielState->MoveNumber() % 3;
}

Key OpenSpielState::hash_key() const
{
    // TODO: Check their method
    std::hash<std::string> hash_string;
    return hash_string(this->fen());
}

void OpenSpielState::flip()
{
    std::cerr << "flip() is unavailable" << std::endl;
}

Action OpenSpielState::uci_to_action(std::string &uciStr) const
{
    // TODO: Write StringToAction
    return spielState->StringToAction(uciStr);
}

std::string OpenSpielState::action_to_san(Action action, const std::vector<Action> &legalActions, bool leadsToWin, bool bookMove) const
{
    // current use UCI move as replacement
    return spielState->ActionToString(spielState->CurrentPlayer(), action);
}

TerminalType OpenSpielState::is_terminal(size_t numberLegalMoves, float &customTerminalValue) const
{
    if (spielState->IsTerminal()) {
	std::cout << "TERMINAL: " << spielState->ToString() << "   -   " << spielState->Returns()[0] << ":" << spielState->Returns()[1] << std::endl;
        const double currentReturn = spielState->Returns()[0];
	std::cout << currentReturn << std::endl;
	if (currentReturn == spielGame->MaxUtility()) {
            return TERMINAL_WIN;
        }
        if (currentReturn == 0) {
            return TERMINAL_DRAW;
        }
        if (currentReturn == spielGame->MinUtility()) {
            return TERMINAL_LOSS;
        }
        customTerminalValue = currentReturn;
        return TERMINAL_CUSTOM;
    }
    return TERMINAL_NONE;
}

bool OpenSpielState::gives_check(Action action) const
{
    // gives_check() is unavailable
    return false;
}

void OpenSpielState::print(std::ostream &os) const
{
    os << spielState->ToString();
}

Tablebase::WDLScore OpenSpielState::check_for_tablebase_wdl(Tablebase::ProbeState &result)
{
    return Tablebase::WDLScoreNone;
}

void OpenSpielState::set_auxiliary_outputs(const float* auxiliaryOutputs)
{
    // do nothing
}

OpenSpielState* OpenSpielState::clone() const
{
    return new OpenSpielState(*this);
}

void OpenSpielState::init(int variant, bool isChess960) {
    check_variant(variant);
    spielState = spielGame->NewInitialState();
    spielState->ApplyAction(1);  // dummy sense action
}

std::string OpenSpielState::ActionToMoveString(Action action, open_spiel::chess::Color actor)
{
    int actorInt = open_spiel::chess::ColorToPlayer(actor);
    std::string moveString = spielState->ActionToString(actorInt,action);        
    return moveString;
}

std::tuple<std::uint8_t,open_spiel::chess::Move,bool> OpenSpielState::ActionToIncompleteMove(Action action, open_spiel::chess::Color actor)
{
    auto result = std::tuple<std::uint8_t,open_spiel::chess::Move,bool>();
    std::uint8_t& phase = std::get<0>(result);
    open_spiel::chess::Move& move = std::get<1>(result);
    bool& passMove = std::get<2>(result);
    
    std::string moveString = ActionToMoveString(action,actor);
    std::string from, to, promotionType;
    auto stringToSquare = [](std::string squareString)
    {
        if(squareString.size()!=2)
            throw std::logic_error("Invalid square string received");
        char col = squareString[0];
        int8_t colInt = col-'a';
        char row = squareString[1];
        int8_t rowInt = row-'1';
        if(colInt<0 || colInt>=8 || rowInt<0 || rowInt>=8)
            throw std::logic_error("Square string contains invalid characters");
        return open_spiel::chess::Square{colInt,rowInt};
    };
    auto stringToPieceType = [](std::string pieceString)
    {
        if(pieceString.size()!=1)
            throw std::logic_error("Invalid piece string received");
        switch (pieceString[0])
        {
            case 'p':
                return open_spiel::chess::PieceType::kPawn;
            case 'n':
                return open_spiel::chess::PieceType::kKnight;
            case 'b':
                return open_spiel::chess::PieceType::kBishop;
            case 'r':
                return open_spiel::chess::PieceType::kRook;
            case 'q':
                return open_spiel::chess::PieceType::kQueen;
            case 'k':
                return open_spiel::chess::PieceType::kKing;
            case ' ':
                return open_spiel::chess::PieceType::kEmpty;
            default:
                throw std::logic_error("Invalid character in piece string");
        }
    };
    
    try
    {
        if(moveString.size()==8)
        {
            phase = 0;
            from = moveString.substr(6,2);
            move.from = stringToSquare(from);
            passMove = false;
        }
        else if(moveString.size()==4 || moveString.size()==5)
        {
            if(moveString=="pass")
            {
                phase = 1;
                from = "a1";
                to = "a1";
                move.from = stringToSquare(from);
                move.to = stringToSquare(to);
                move.promotion_type = open_spiel::chess::PieceType::kEmpty;
                passMove = true;
            }
            else
            {
                phase = 1;
                from = moveString.substr(0,2);
                to = moveString.substr(2,2);
                move.from = stringToSquare(from);
                move.to = stringToSquare(to);
                move.promotion_type = open_spiel::chess::PieceType::kEmpty;
                passMove = false;
                if(moveString.size()==5)
                {
                    promotionType = moveString.substr(4,1);
                    move.promotion_type = stringToPieceType(promotionType);
                }
            }
        }
        else
            throw std::logic_error("Invalid LAN string received: "+std::to_string(moveString.size())+"|"+moveString+"|");
    }
    catch(std::logic_error e)
    {
        std::cout<<"What:"<<e.what()<<std::endl;
        throw std::logic_error("Problem: "+moveString);
    }     
    return result;
}

open_spiel::chess::Color OpenSpielState::currentPlayer()
{
    return open_spiel::chess::PlayerToColor(spielState->CurrentPlayer());
}

open_spiel::rbc::MovePhase OpenSpielState::currentPhase() const
{
    open_spiel::rbc::RbcState* rbcState = dynamic_cast<open_spiel::rbc::RbcState*>(spielState.get());
    if(rbcState==nullptr)
        throw std::logic_error("Can not cast state to RbcState");
    return rbcState->phase();
}

std::string OpenSpielState::ToString()
{
    open_spiel::rbc::RbcState* rbcState = dynamic_cast<open_spiel::rbc::RbcState*>(spielState.get());
    if(rbcState==nullptr)
        throw std::logic_error("Cast to RBC state failed");
    return rbcState->Board().ToFEN();
}


