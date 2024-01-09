#include "chessinformationset.h"

namespace crazyara {

ChessInformationSet::ChessInformationSet
()
:InformationSet()
{}

ChessInformationSet::PieceType ChessInformationSet::OpenSpielPieceType_to_CISPieceType
(
    const open_spiel::chess::PieceType os_pT
)
{
    PieceType result;
    switch(os_pT)
    {
        case open_spiel::chess::PieceType::kPawn:
            result = PieceType::pawn;
            break;
        case open_spiel::chess::PieceType::kKnight:
            result = PieceType::knight;
            break;        
        case open_spiel::chess::PieceType::kBishop:
            result = PieceType::bishop;
            break;        
        case open_spiel::chess::PieceType::kRook:
            result = PieceType::rook;
            break;        
        case open_spiel::chess::PieceType::kQueen:
            result = PieceType::queen;
            break;        
        case open_spiel::chess::PieceType::kKing:
            result = PieceType::king;
            break;
        case open_spiel::chess::PieceType::kEmpty:
            result = PieceType::empty;
            break;
        default:
            throw std::logic_error("Conversion failure from open_spiel::chess::PieceType to ChessInformationSet::PieceType!");
    }
    return result;
}

open_spiel::chess::PieceType ChessInformationSet::CISPieceType_to_OpenSpielPieceType
(
    const ChessInformationSet::PieceType cis_pT
)
{
    open_spiel::chess::PieceType result;
    switch(cis_pT)
    {
        case PieceType::pawn:
            result = open_spiel::chess::PieceType::kPawn;
            break;
        case PieceType::knight:
            result = open_spiel::chess::PieceType::kKnight;
            break;        
        case PieceType::bishop:
            result = open_spiel::chess::PieceType::kBishop;
            break;        
        case PieceType::rook:
            result = open_spiel::chess::PieceType::kRook;
            break;        
        case PieceType::queen:
            result = open_spiel::chess::PieceType::kQueen;
            break;        
        case PieceType::king:
            result = open_spiel::chess::PieceType::kKing;
            break;
        case PieceType::empty:
            result = open_spiel::chess::PieceType::kEmpty;
            break;
        default:
            throw std::logic_error("Conversion failure from ChessInformationSet::PieceType to open_spiel::chess::PieceType!");
    }
    return result;
}

std::function<bool(const ChessInformationSet::Square&)> ChessInformationSet::OnePlayerChessInfo::getBlockCheck()
{
    squareToPieceMap.clear();
    auto insert = [&](const std::vector<ChessInformationSet::Square>& pieces, const PieceType pT)
    {
        for(const ChessInformationSet::Square& sq : pieces)
        {
            squareToPieceMap[sq] = pT;
        }
    };
    
    insert(pawns,ChessInformationSet::PieceType::pawn);
    insert(knights,ChessInformationSet::PieceType::knight);
    insert(bishops,ChessInformationSet::PieceType::bishop);
    insert(rooks,ChessInformationSet::PieceType::rook);
    insert(queens,ChessInformationSet::PieceType::queen);
    insert(kings,ChessInformationSet::PieceType::king);
    
    return [&](const Square& sq){return this->squareToPieceMap.find(sq)!=this->squareToPieceMap.end();};
}

std::function<bool(const ChessInformationSet::Square&)> ChessInformationSet::OnePlayerChessInfo::getBlockCheck
(
    const std::vector<ChessInformationSet::Square>& onePieceType,
    const PieceType pT
)
{
    squareToPieceMap.clear();
    for(const ChessInformationSet::Square& sq : onePieceType)
    {
        squareToPieceMap[sq] = pT; //Insert random piece
    }

    return [&](const Square& sq){return this->squareToPieceMap.find(sq)!=this->squareToPieceMap.end();};
}

std::function<std::pair<bool,ChessInformationSet::PieceType>(const ChessInformationSet::Square&)> ChessInformationSet::OnePlayerChessInfo::getSquarePieceTypeCheck()
{
    squareToPieceMap.clear();
    auto insert = [&](const std::vector<ChessInformationSet::Square>& pieces, const PieceType pT)
    {
        for(const ChessInformationSet::Square& sq : pieces)
        {
            squareToPieceMap[sq] = pT;
        }
    };
    
    insert(pawns,ChessInformationSet::PieceType::pawn);
    insert(knights,ChessInformationSet::PieceType::knight);
    insert(bishops,ChessInformationSet::PieceType::bishop);
    insert(rooks,ChessInformationSet::PieceType::rook);
    insert(queens,ChessInformationSet::PieceType::queen);
    insert(kings,ChessInformationSet::PieceType::king);
    
    auto squarePieceTypeCheck = [&](const Square& sq)
    {
        std::pair<bool,PieceType> pt;
        auto iter = this->squareToPieceMap.find(sq);
        if(iter!=this->squareToPieceMap.end())
        {
            pt.first = true;
            pt.second = iter->second;
        }
        else
        {
            pt.first = false;
        }
        return pt;
    };
    
    return squarePieceTypeCheck;
}

std::function<std::vector<ChessInformationSet::Square>::iterator(const ChessInformationSet::Square&)> ChessInformationSet::OnePlayerChessInfo::getPieceIter
(
    std::vector<ChessInformationSet::Square>& onePieceType
)
{
    squareToPieceIter.clear();
    for(auto iter=onePieceType.begin(); iter!=onePieceType.end(); iter++)
    {
        squareToPieceIter[*iter] = iter;
    }

    return [&](const Square& sq)
        {
            auto iter = this->squareToPieceIter.find(sq);
            if(iter==this->squareToPieceIter.end())
                return onePieceType.end();
            else
                return iter->second;
        };
}

void ChessInformationSet::setBoard
(
    ChessInformationSet::OnePlayerChessInfo& pieces,
    double probability,
    std::uint64_t index
)
{
    std::unique_ptr<std::bitset<chessInfoSize>> board = encodeBoard(pieces,probability);;
        
    setBitPattern(index,*board);
}

std::unique_ptr<std::pair<ChessInformationSet::OnePlayerChessInfo,double>> ChessInformationSet::getBoard
(
    const std::uint64_t index
) const
{
    std::unique_ptr<std::pair<OnePlayerChessInfo,double>> board;
    
    std::unique_ptr<std::bitset<chessInfoSize>> bits = getBitPattern(index);
    std::bitset<chessInfoSize> bitPattern = *bits;
    
    board = decodeBoard(bitPattern);
    
    return board;
}

std::unique_ptr<std::pair<ChessInformationSet::OnePlayerChessInfo,double>> ChessInformationSet::decodeBoard
(
    const std::bitset<chessInfoSize>& bits
) const
{
    auto board = std::make_unique<std::pair<OnePlayerChessInfo,double>>();
    
    std::vector<std::vector<ChessInformationSet::Square>*> pieces = 
        {&(board->first.pawns),
         &(board->first.knights),
         &(board->first.bishops),
         &(board->first.rooks),
         &(board->first.queens),
         &(board->first.kings)};
    
    std::uint16_t bitStartInd = 0;    
    for(std::uint8_t pieceType=0; pieceType<6; pieceType++)
    {
        for(std::uint8_t boardInd=0; boardInd<64; boardInd++)
        {
            std::uint16_t bitInd = bitStartInd+boardInd;
            if(bits[bitInd])
            {
                pieces[pieceType]->push_back(boardIndexToSquare(boardInd));
            }
        }
        bitStartInd += 64;
    }
    
    board->first.kingside = bits[bitStartInd];
    bitStartInd++;
    
    board->first.queenside = bits[bitStartInd];
    bitStartInd++;
    
    for(std::uint8_t boardInd=0; boardInd<64; boardInd++)
    {
        std::uint16_t bitInd = bitStartInd+boardInd;
        if(bits[bitInd])
        {
            board->first.en_passant.push_back(boardIndexToSquare(boardInd));
        }
    }
    bitStartInd += 64;
    
    std::uint8_t& no_progress_count = board->first.no_progress_count;
    no_progress_count = transferBitPattern<std::uint8_t>(bits,bitStartInd,7);
    bitStartInd += 7;
    
    std::uint8_t probabilityInt = transferBitPattern<std::uint8_t>(bits,bitStartInd,7);
    std::bitset<7> probabilityIntBitMax;
    probabilityIntBitMax.flip();
    unsigned long probabilityIntMax = probabilityIntBitMax.to_ulong();
    double probability = probabilityInt;
    probability /= static_cast<double>(probabilityIntMax);
    board->second = probability;
    
    return board;
}

double ChessInformationSet::Distribution::getProbability
(
    const ChessInformationSet::Square& sq,
    const ChessInformationSet::PieceType pT
) const
{
    return getProbability(squareToBoardIndex(sq),pT);
}

double ChessInformationSet::Distribution::getProbability
(
    const std::uint8_t sqInd,
    const ChessInformationSet::PieceType pT
) const
{
    switch (pT)
    {
        case PieceType::pawn:
            return pawns[sqInd];
        case PieceType::knight:
            return knights[sqInd];
        case PieceType::bishop:
            return bishops[sqInd];
        case PieceType::rook:
            return rooks[sqInd];
        case PieceType::queen:
            return queens[sqInd];
        case PieceType::king:
            return kings[sqInd];
        default:
            double prob = 1.0;
            prob -= pawns[sqInd];
            prob -= knights[sqInd];
            prob -= bishops[sqInd];
            prob -= rooks[sqInd];
            prob -= queens[sqInd];
            prob -= kings[sqInd];
            return prob;
    }        
}
        
std::unique_ptr<std::bitset<chessInfoSize>> ChessInformationSet::encodeBoard
(
    ChessInformationSet::OnePlayerChessInfo& piecesInfo,
    double probability
) const
{
    using CIS=ChessInformationSet;
    
    auto bits = std::make_unique<std::bitset<chessInfoSize>>();
    std::bitset<chessInfoSize>& bitBoard = *bits;

    std::vector<std::vector<CIS::Square>*> pieces = 
        {&(piecesInfo.pawns),
         &(piecesInfo.knights),
         &(piecesInfo.bishops),
         &(piecesInfo.rooks),
         &(piecesInfo.queens),
         &(piecesInfo.kings)};
    
    std::uint16_t bitStartInd = 0;    
    for(std::uint8_t pieceType=0; pieceType<6; pieceType++)
    {
        std::function<bool(Square)> pieceCheck = piecesInfo.getBlockCheck(*(pieces[pieceType]),static_cast<CIS::PieceType>(pieceType));
        for(std::uint8_t boardInd=0; boardInd<64; boardInd++)
        {
            CIS::Square sq = boardIndexToSquare(boardInd);
            if(pieceCheck(sq))                
            {
                std::uint16_t bitInd = bitStartInd+boardInd;
                bitBoard[bitInd] = true;
            }
        }
        bitStartInd += 64;
    }
    
    bitBoard[bitStartInd] = piecesInfo.kingside;
    bitStartInd++;
    
    bitBoard[bitStartInd] = piecesInfo.queenside;
    bitStartInd++;
    
    std::function<bool(Square)> pieceCheck = piecesInfo.getBlockCheck(piecesInfo.en_passant,static_cast<CIS::PieceType>(0));
    for(std::uint8_t boardInd=0; boardInd<64; boardInd++)
    {
        CIS::Square sq = boardIndexToSquare(boardInd);
        if(pieceCheck(sq))                
        {
            std::uint16_t bitInd = bitStartInd+boardInd;
            bitBoard[bitInd] = true;
        }
    }
    bitStartInd += 64;
    
    std::uint8_t no_progress_count = piecesInfo.no_progress_count;
    if(no_progress_count>127)
        throw std::logic_error("no_progress_count must not be larger than 100");
    assignBitPattern<std::uint8_t>(bitBoard,bitStartInd,no_progress_count,7);
    bitStartInd += 7;
    
    std::bitset<7> probabilityIntBitMax;
    probabilityIntBitMax.flip();
    double probabilityMax = static_cast<double>(probabilityIntBitMax.to_ulong());
    if(probability<0 || probability>1)
        throw std::logic_error("probability must be inside [0,1]");
    probability = probability*probabilityMax;
    std::uint8_t probabilityInt = static_cast<std::uint8_t>(probability);
    assignBitPattern<std::uint8_t>(bitBoard,bitStartInd,probabilityInt,7);

    return bits;
}

std::unique_ptr<ChessInformationSet::Distribution> ChessInformationSet::computeDistribution()
{
    using CIS = ChessInformationSet;
    
    auto cis_distribution = std::make_unique<Distribution>();
    std::uint64_t numberOfBoards = 0;
    for(auto iter = this->begin(); iter!=this->end(); iter++,numberOfBoards++)
    {
        std::unique_ptr<std::pair<OnePlayerChessInfo,double>> oneBoard = *iter;
        double probability = oneBoard->second;
        const OnePlayerChessInfo& pieceData = oneBoard->first;
        
        std::array<double,64>& pawnDist = cis_distribution->pawns;
        for(const CIS::Square& sq : pieceData.pawns)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            pawnDist[index] += 1.0;
        }

        std::array<double,64>& knightDist = cis_distribution->knights;
        for(const CIS::Square& sq : pieceData.knights)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            knightDist[index] += 1.0;
        }

        std::array<double,64>& bishopDist = cis_distribution->bishops;
        for(const CIS::Square& sq : pieceData.bishops)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            bishopDist[index] += 1.0;
        }

        std::array<double,64>& rookDist = cis_distribution->rooks;
        for(const CIS::Square& sq : pieceData.rooks)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            rookDist[index] += 1.0;
        }

        std::array<double,64>& queenDist = cis_distribution->queens;
        for(const CIS::Square& sq : pieceData.queens)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            queenDist[index] += 1.0;
        }
        
        std::array<double,64>& kingDist = cis_distribution->kings;
        for(const CIS::Square& sq : pieceData.kings)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            kingDist[index] += 1.0;
        }
        
        if(pieceData.kingside)
            cis_distribution->kingside += 1.0;

        if(pieceData.queenside)
            cis_distribution->queenside += 1.0;
        
        std::array<double,64>& en_passantDist = cis_distribution->en_passant;
        for(const CIS::Square& sq : pieceData.en_passant)
        {
            unsigned int index = CIS::squareToBoardIndex(sq);
            en_passantDist[index] += 1.0;
        }
        
        cis_distribution->no_progress_count += pieceData.no_progress_count;
    }
    
    auto divideBoard = [](std::array<double,64> board, std::uint64_t value)
    {
        for(double& count : board)
            count /= value;
    };
    
    divideBoard(cis_distribution->pawns,numberOfBoards);
    divideBoard(cis_distribution->knights,numberOfBoards);
    divideBoard(cis_distribution->bishops,numberOfBoards);
    divideBoard(cis_distribution->rooks,numberOfBoards);
    divideBoard(cis_distribution->queens,numberOfBoards);
    divideBoard(cis_distribution->kings,numberOfBoards);
    cis_distribution->kingside /= numberOfBoards;
    cis_distribution->queenside /= numberOfBoards;
    divideBoard(cis_distribution->en_passant,numberOfBoards);   
    cis_distribution->no_progress_count /= numberOfBoards;
    
    return cis_distribution;    
}


void ChessInformationSet::add
(
    ChessInformationSet::OnePlayerChessInfo& item,
    double probability
)
{
    std::unique_ptr<std::bitset<chessInfoSize>> bits = encodeBoard(item,probability);
    InformationSet<chessInfoSize>::add(*bits);
};
        
void ChessInformationSet::add
(
    std::vector<std::pair<ChessInformationSet::OnePlayerChessInfo,double>>& items
)
{
    std::vector<std::bitset<chessInfoSize>> encodedItems(items.size());
    for(uint i=0; i<items.size(); i++)
    {
        encodedItems[i] = *encodeBoard(items[i].first,items[i].second);
    }
    InformationSet<chessInfoSize>::add(encodedItems);
}

void ChessInformationSet::markIncompatibleBoards
(
    const std::vector<BoardClause>& conditions
)
{
    std::cout<<"Mark boards that do not fit: ";
    for(auto clause : conditions)
        std::cout<<clause.to_string()<<"&&";
    std::cout<<std::endl;
    std::unique_ptr<std::pair<OnePlayerChessInfo,double>> board;
    for(auto iter=begin(); iter!=end(); iter++)
    {
        board = *iter;
        OnePlayerChessInfo& piecesInfo = board->first;
        
        if(!evaluateHornClause(conditions,piecesInfo))
            incompatibleBoards.push(iter.getCurrentIndex());
    }
}

bool ChessInformationSet::evaluateHornClause
(
    const std::vector<BoardClause>& hornClause,
    OnePlayerChessInfo& piecesInfo
)
{
    bool value = true;
    for(const BoardClause& oneClause : hornClause)
    {
        value = value && oneClause(piecesInfo);
    }
    return value;
}

void ChessInformationSet::removeIncompatibleBoards()
{
    while(!incompatibleBoards.empty())
    {
        std::uint64_t ind = incompatibleBoards.front();
        InformationSet<chessInfoSize>::remove(ind);
        incompatibleBoards.pop();
    }
}

ChessInformationSet::Square::Square()
{}

ChessInformationSet::Square::Square
(
    ChessInformationSet::ChessColumn column,
    ChessInformationSet::ChessRow row
):
column(column),
row(row)
{}

ChessInformationSet::Square::Square
(
    std::uint8_t column,
    std::uint8_t row
)
{
    if(column<0 || column>=8)
        std::logic_error("Column integer out of bounds");
    this->column = static_cast<ChessInformationSet::ChessColumn>(column);
    if(row<0 || row>=8)
        std::logic_error("Row integer out of bounds");
    this->row = static_cast<ChessInformationSet::ChessRow>(row);    
}

ChessInformationSet::Square::Square(const open_spiel::chess_common::Square& os_sq)
{
    if(os_sq.x<0 || os_sq.x>=8 || os_sq.y<0 || os_sq.y>=8)
        throw std::logic_error("open_spiel::chess_common::Square out of board bounds");
    std::uint8_t ux = os_sq.x;
    std::uint8_t uy = os_sq.y;
    
    column = static_cast<ChessColumn>(ux);
    row = static_cast<ChessRow>(uy);
}

bool ChessInformationSet::Square::vertPlus(std::uint8_t multiple) {return moveSquare(0,multiple);}
bool ChessInformationSet::Square::vertMinus(std::uint8_t multiple){return moveSquare(0,-multiple);}

bool ChessInformationSet::Square::horizPlus(std::uint8_t multiple){return moveSquare(multiple,0);}
bool ChessInformationSet::Square::horizMinus(std::uint8_t multiple){return moveSquare(-multiple,0);}

bool ChessInformationSet::Square::diagVertPlusHorizPlus(std::uint8_t multiple){return moveSquare(multiple,multiple);}
bool ChessInformationSet::Square::diagVertMinusHorizPlus(std::uint8_t multiple){return moveSquare(multiple,-multiple);}
bool ChessInformationSet::Square::diagVertPlusHorizMinus(std::uint8_t multiple){return moveSquare(-multiple,multiple);}
bool ChessInformationSet::Square::diagVertMinusHorizMinus(std::uint8_t multiple){return moveSquare(-multiple,-multiple);}

/*
bool ChessInformationSet::Square::knightVertPlusHorizPlus(){return moveSquare(1,2);}
bool ChessInformationSet::Square::knightVertPlusHorizMinus(){return moveSquare(-1,2);}
bool ChessInformationSet::Square::knightVertMinusHorizPlus(){return moveSquare(1,-2);}
bool ChessInformationSet::Square::knightVertMinusHorizMinus(){return moveSquare(-1,-2);}
bool ChessInformationSet::Square::knightHorizPlusVertPlus(){return moveSquare(2,1);}
bool ChessInformationSet::Square::knightHorizPlusVertMinus(){return moveSquare(2,-1);}
bool ChessInformationSet::Square::knightHorizMinusVertPlus(){return moveSquare(-2,1);}
bool ChessInformationSet::Square::knightHorizMinusVertMinus(){return moveSquare(-2,-1);}
*/

bool ChessInformationSet::Square::validSquare(std::int8_t column, std::int8_t row)
{
    return (column>=0 && column<8 && row>=0 && row<8);
}

bool ChessInformationSet::Square::moveSquare(std::int8_t deltaCol, std::int8_t deltaRow)
{
    std::int8_t col = static_cast<std::int8_t>(this->column);
    std::int8_t row = static_cast<std::int8_t>(this->row);

    col+=deltaCol;
    row+=deltaRow;
    
    if(!validSquare(col,row))
        return false;
    
    std::uint8_t ucol = static_cast<std::uint8_t>(col);
    std::uint8_t urow = static_cast<std::uint8_t>(row);
    
    this->column = static_cast<ChessColumn>(ucol);
    this->row = static_cast<ChessRow>(urow);
    return true;
}

std::pair<std::int8_t,std::int8_t> ChessInformationSet::Square::diffToSquare(const ChessInformationSet::Square& sq)
{
    std::int8_t col = static_cast<std::int8_t>(this->column);
    std::int8_t row = static_cast<std::int8_t>(this->row);
    std::int8_t sqCol = static_cast<std::int8_t>(sq.column);
    std::int8_t sqRow = static_cast<std::int8_t>(sq.row);
    
    std::int8_t deltaCol = sqCol - col; 
    std::int8_t deltaRow = sqRow - row;
    
    return {deltaCol,deltaRow};
}
}
