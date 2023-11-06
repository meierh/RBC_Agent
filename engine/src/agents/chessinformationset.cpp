#include "chessinformationset.h"

namespace crazyara {

ChessInformationSet::ChessInformationSet
()
:InformationSet()
{}

ChessInformationSet::PieceType ChessInformationSet::boardIndexToPieceType
(
    std::uint8_t boardIndex
)
{
    assert(boardIndex<16);
    PieceType type;    
    if(boardIndex<8)
        type = PieceType::pawn;
    else if(boardIndex==8 || boardIndex==15)
        type = PieceType::rook;
    else if(boardIndex==9 || boardIndex==14)
        type = PieceType::knight;
    else if(boardIndex==10 || boardIndex==13)
        type = PieceType::bishop;
    else if(boardIndex==11)
        type = PieceType::queen;
    else if(boardIndex==12)
        type = PieceType::king;
    else
        assert(false);
    return type;
}

std::unique_ptr<std::array<std::pair<ChessInformationSet::Square,bool>,8>> ChessInformationSet::ChessPiecesInformation::extractPawns() const
{
    auto res = std::make_unique<std::array<std::pair<Square,bool>,8>>();
    for(unsigned int i=0;i<8;i++)
        (*res)[i] = data[i];
    return res;
};

std::unique_ptr<std::array<std::pair<ChessInformationSet::Square,bool>,2>> ChessInformationSet::ChessPiecesInformation::extractRooks() const
{
    auto res = std::make_unique<std::array<std::pair<Square,bool>,2>>();
    (*res)[0] = data[8];
    (*res)[1] = data[15];
    return res;
};

std::unique_ptr<std::array<std::pair<ChessInformationSet::Square,bool>,2>> ChessInformationSet::ChessPiecesInformation::extractKnights() const
{
    auto res = std::make_unique<std::array<std::pair<Square,bool>,2>>();
    (*res)[0] = data[9];
    (*res)[1] = data[14];
    return res;
};

std::unique_ptr<std::array<std::pair<ChessInformationSet::Square,bool>,2>> ChessInformationSet::ChessPiecesInformation::extractBishops() const
{
    auto res = std::make_unique<std::array<std::pair<Square,bool>,2>>();
    (*res)[0] = data[10];
    (*res)[1] = data[13];
    return res;
};

std::unique_ptr<std::array<std::pair<ChessInformationSet::Square,bool>,1>> ChessInformationSet::ChessPiecesInformation::extractQueens() const
{
    auto res = std::make_unique<std::array<std::pair<Square,bool>,1>>();
    (*res)[0] = data[11];
    return res;
};

std::unique_ptr<std::array<std::pair<ChessInformationSet::Square,bool>,1>> ChessInformationSet::ChessPiecesInformation::extractKings() const
{
    auto res = std::make_unique<std::array<std::pair<Square,bool>,1>>();
    (*res)[0] = data[12];
    return res;
};

std::function<bool(const ChessInformationSet::Square&)> ChessInformationSet::ChessPiecesInformation::getBlockCheck()
{
    squareToPiece.clear();
    for(std::uint8_t i=0; i<data.size(); i++)
    {
        const std::pair<Square,bool>& piece = data[i];
        if(piece.second)
        {
            squareToPiece[piece.first] = static_cast<Piece>(i);
        }
    }
    return [&](const Square& sq){return this->squareToPiece.find(sq)!=this->squareToPiece.end();};
}


void ChessInformationSet::setBoard
(
    const ChessInformationSet::ChessPiecesInformation& pieces,
    const double probability,
    const std::uint64_t index
)
{
    std::unique_ptr<std::bitset<chessInfoSize>> board = encodeBoard(pieces,probability);;
        
    setBitPattern(index,*board);
}

std::unique_ptr<std::pair<ChessInformationSet::ChessPiecesInformation,double>> ChessInformationSet::getBoard
(
    const std::uint64_t index
) const
{
    std::unique_ptr<std::pair<ChessPiecesInformation,double>> board;
    
    std::unique_ptr<std::bitset<chessInfoSize>> bits = getBitPattern(index);
    std::bitset<chessInfoSize> bitPattern = *bits;
    
    board = decodeBoard(bitPattern);
    
    return board;
}

std::unique_ptr<std::pair<ChessInformationSet::ChessPiecesInformation,double>> ChessInformationSet::decodeBoard
(
    const std::bitset<chessInfoSize>& bits
) const
{
    auto board = std::make_unique<std::pair<ChessPiecesInformation,double>>();
        
    std::uint16_t prob16bit = transferBitPattern<std::uint16_t>(bits,0,16);
    double probability = prob16bit;
    probability /= std::numeric_limits<std::uint16_t>::max();
    board->second = probability;
    
    std::array<std::pair<Square,bool>,16>& pieces = board->first.data;

    std::uint8_t bitInd=16;   
    std::uint8_t pieceInd=0;
    for(; pieceInd<16; pieceInd++, bitInd+=7)
    {
        pieces[pieceInd].second = bits[bitInd];
        pieces[pieceInd].first.column = static_cast<ChessColumn>(transferBitPattern<std::uint64_t>(bits,bitInd+1,3));
        pieces[pieceInd].first.row = static_cast<ChessRow>(transferBitPattern<std::uint64_t>(bits,bitInd+4,3));
    }
    
    return board;
}
        
std::unique_ptr<std::bitset<chessInfoSize>> ChessInformationSet::encodeBoard
(
    const ChessInformationSet::ChessPiecesInformation& piecesInfo,
    const double probability
) const
{
    auto bits = std::make_unique<std::bitset<chessInfoSize>>();
    std::bitset<chessInfoSize>& bitBoard = *bits;
    
    std::uint16_t prob16bit = std::numeric_limits<std::uint16_t>::max()*probability;
    std::uint8_t bitInd=0;
    for(;bitInd<16;bitInd++)
    {
        bitBoard[bitInd] = getBit<std::uint16_t>(prob16bit,15-bitInd);
    }
    
    const std::array<std::pair<Square,bool>,16>& pieces = piecesInfo.data;
    
    std::uint8_t pieceInd=0;
    for(; pieceInd<16; pieceInd++, bitInd+=7)
    {
        if(pieces[pieceInd].second)
        {
            bitBoard[bitInd] = true;
            std::uint8_t column = static_cast<std::uint8_t>(pieces[pieceInd].first.column);
            assignBitPattern(bitBoard,bitInd+1,column,3);
            std::uint8_t row = static_cast<std::uint8_t>(pieces[pieceInd].first.row);
            assignBitPattern(bitBoard,bitInd+4,row,3);
        }
        else
        {
            bitBoard[bitInd] = false;
            std::uint8_t zeros = 0;
            assignBitPattern(bitBoard,bitInd+1,zeros,6);
        }
    }
    return bits;
}

void ChessInformationSet::add
(
    const ChessInformationSet::ChessPiecesInformation& item,
    double probability
)
{
    std::unique_ptr<std::bitset<chessInfoSize>> bits = encodeBoard(item,probability);
    InformationSet<chessInfoSize>::add(*bits);
};
        
void ChessInformationSet::add
(
    const std::vector<std::pair<ChessInformationSet::ChessPiecesInformation,double>>& items
)
{
    for(const std::pair<ChessPiecesInformation,double>& item : items)
        add(item.first,item.second);
}

void ChessInformationSet::markIncompatibleBoards
(
    std::vector<Square>& noPieces,
    std::vector<Square>& unknownPieces,
    std::vector<std::pair<PieceType,Square>>& knownPieces
)
{
    auto squareMatch = [&](const ChessPiecesInformation& piecesInfo, const Square& sq)
    {
        const std::array<std::pair<Square,bool>,16>& pieces = piecesInfo.data;
        std::pair<bool,PieceType> match;
        for(std::uint8_t pieceInd=0; pieceInd<pieces.size(); pieceInd++)
        {
            const std::pair<Square,bool>& piece = pieces[pieceInd];
            if(piece.second)
            {
                if(piece.first==sq)
                {
                    assert(!match.first);
                    match.second = boardIndexToPieceType(pieceInd);
                }
            }
        }
        return match;
    };
    
    std::unique_ptr<std::pair<ChessPiecesInformation,double>> board;
    for(std::uint64_t infoSetIndex = 0; infoSetIndex<this->size(); infoSetIndex++)
    {
        bool incompatibleBoard = false;
        board = getBoard(infoSetIndex);
        const ChessPiecesInformation& piecesInfo = board->first;
        
        for(const Square& sq : noPieces)
        {
            if(squareMatch(piecesInfo,sq).first)
            {
                incompatibleBoard=true;
            }
        }        
        for(const Square& sq : unknownPieces)
        {
            if(!squareMatch(piecesInfo,sq).first)
            {
                incompatibleBoard=true;
            }
        }
        for(const std::pair<PieceType,Square>& pieceSq : knownPieces)
        {
            const Square& sq = pieceSq.second;
            const PieceType& pieceType = pieceSq.first;
            std::pair<bool,PieceType> match = squareMatch(piecesInfo,sq);
            if(!match.first || match.second!=pieceType)
            {
                incompatibleBoard=true;
            }
        }
        incompatibleBoards.push(infoSetIndex);
    }
}

bool ChessInformationSet::Square::vertPlus(std::uint8_t multiple) {return moveSquare(0,multiple);}
bool ChessInformationSet::Square::vertMinus(std::uint8_t multiple){return moveSquare(0,-multiple);}

bool ChessInformationSet::Square::horizPlus(std::uint8_t multiple){return moveSquare(multiple,0);}
bool ChessInformationSet::Square::horizMinus(std::uint8_t multiple){return moveSquare(-multiple,0);}

bool ChessInformationSet::Square::diagVertPlusHorizPlus(std::uint8_t multiple){return moveSquare(multiple,multiple);}
bool ChessInformationSet::Square::diagVertMinusHorizPlus(std::uint8_t multiple){return moveSquare(multiple,-multiple);}
bool ChessInformationSet::Square::diagVertPlusHorizMinus(std::uint8_t multiple){return moveSquare(-multiple,multiple);}
bool ChessInformationSet::Square::diagVertMinusHorizMinus(std::uint8_t multiple){return moveSquare(-multiple,-multiple);}

bool ChessInformationSet::Square::knightVertPlusHorizPlus(){return moveSquare(1,2);}
bool ChessInformationSet::Square::knightVertPlusHorizMinus(){return moveSquare(-1,2);}
bool ChessInformationSet::Square::knightVertMinusHorizPlus(){return moveSquare(1,-2);}
bool ChessInformationSet::Square::knightVertMinusHorizMinus(){return moveSquare(-1,-2);}
bool ChessInformationSet::Square::knightHorizPlusVertPlus(){return moveSquare(2,1);}
bool ChessInformationSet::Square::knightHorizPlusVertMinus(){return moveSquare(2,-1);}
bool ChessInformationSet::Square::knightHorizMinusVertPlus(){return moveSquare(-2,1);}
bool ChessInformationSet::Square::knightHorizMinusVertMinus(){return moveSquare(-2,-1);}

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
}
