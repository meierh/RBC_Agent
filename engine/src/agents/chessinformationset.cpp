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

void ChessInformationSet::setBoard
(
    const std::array<std::pair<Square,bool>,16> pieces,
    const double probability,
    const std::uint64_t index
)
{
    std::unique_ptr<std::bitset<chessInfoSize>> board = encodeBoard(pieces,probability);;
        
    setBitPattern(index,*board);
}

std::unique_ptr<std::pair<std::array<std::pair<ChessInformationSet::Square,bool>,16>,double>> ChessInformationSet::getBoard
(
    const std::uint64_t index
) const
{
    std::unique_ptr<std::pair<std::array<std::pair<Square,bool>,16>,double>> board;
    
    std::unique_ptr<std::bitset<chessInfoSize>> bits = getBitPattern(index);
    std::bitset<chessInfoSize> bitPattern = *bits;
    
    board = decodeBoard(bitPattern);
    
    return board;
}

std::unique_ptr<std::pair<std::array<std::pair<ChessInformationSet::Square,bool>,16>,double>> ChessInformationSet::decodeBoard
(
    const std::bitset<chessInfoSize>& bits
) const
{
    auto board = std::make_unique<std::pair<std::array<std::pair<Square,bool>,16>,double>>();
        
    std::uint16_t prob16bit = transferBitPattern<std::uint16_t>(bits,0,16);
    double probability = prob16bit;
    probability /= std::numeric_limits<std::uint16_t>::max();
    board->second = probability;
    
    std::array<std::pair<Square,bool>,16>& pieces = board->first;

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
    const std::array<std::pair<ChessInformationSet::Square,bool>,16>& pieces,
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

void ChessInformationSet::markIncompatibleBoards
(
    std::vector<Square>& noPieces,
    std::vector<Square>& unknownPieces,
    std::vector<std::pair<PieceType,Square>>& knownPieces
)
{
    auto squareMatch = [&](const std::array<std::pair<Square,bool>,16>& pieces, const Square& sq)
    {
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
    
    std::unique_ptr<std::pair<std::array<std::pair<Square,bool>,16>,double>> board;
    for(std::uint64_t infoSetIndex = 0; infoSetIndex<this->size(); infoSetIndex++)
    {
        bool incompatibleBoard = false;
        board = getBoard(infoSetIndex);
        const std::array<std::pair<Square,bool>,16>& pieces = board->first;
        
        for(const Square& sq : noPieces)
        {
            if(squareMatch(pieces,sq).first)
            {
                incompatibleBoard=true;
            }
        }        
        for(const Square& sq : unknownPieces)
        {
            if(!squareMatch(pieces,sq).first)
            {
                incompatibleBoard=true;
            }
        }
        for(const std::pair<PieceType,Square>& pieceSq : knownPieces)
        {
            const Square& sq = pieceSq.second;
            const PieceType& pieceType = pieceSq.first;
            std::pair<bool,PieceType> match = squareMatch(pieces,sq);
            if(!match.first || match.second!=pieceType)
            {
                incompatibleBoard=true;
            }
        }
        incompatibleBoards.push(infoSetIndex);
    }
}

}
