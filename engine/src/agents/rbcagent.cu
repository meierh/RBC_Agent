#include "rbcagent.h"
#include "chessinformationset.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace crazyara {

__device__ bool RBCAgetBit
(
    uint8_t byte,
    uint8_t position
)
{
    return byte & (1 << position);
}
    
__device__ void numberToChar
(
    char** string,
    uint16_t number 
)
{
    bool startedWriting = false;
    if(number==0)
    {
        **string = '0';
        *string = *string + 1;
        return;
    }
    for(uint16_t decimal = 10000; decimal>0; decimal /= 10)
    {
        uint16_t firstDigit = number / decimal;
        uint16_t remainingDigits = number % decimal;
        **string = '0'+firstDigit;
        if(firstDigit!=0)
        {
            *string = *string + 1;
        }
        else
        {
            if(startedWriting)
                *string = *string + 1;
        }
        startedWriting = startedWriting || (firstDigit!=0);
        number = remainingDigits;
    }
}

__global__ void genFEN
(
    
    uint8_t* oppoBoardInfoSet, // 6*64 pieces + 2*1 castling + 64 en_passant + 7 halfmove + 7 prob (in bits)
    char oppoPieceCharSet[6], // pawn=0,knight=1,bishop=2,rook=3,queen=4,king=5
    char selfColor, // w or b
    uint8_t* selfBoardInfoSet, // 6*64 pieces + 2*1 castling + 64 en_passant + 7 halfmove + 7 prob (in bits)
    char selfPieceCharSet[6], // pawn=0,knight=1,bishop=2,rook=3,queen=4,king=5
    char turnColor, // w or b
    uint64_t boardInfoSetSize,
    uint16_t nextMoveNumber,
    char* fenVector // 100 byte per Item    
)
{   
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ uint8_t selfBoard[58];
    __shared__ char selfPiecesChar[6];
    __shared__ char oppoPiecesChar[6];
    if(threadIdx.x==0)
    {
        for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
        {
            oppoPiecesChar[pieceInd] = oppoPieceCharSet[pieceInd];
            selfPiecesChar[pieceInd] = selfPieceCharSet[pieceInd];
        }
        for(uint8_t byteInd=0; byteInd<58; byteInd++)
        {
            selfBoard[byteInd] = selfBoardInfoSet[byteInd];
        }
    }
    __syncthreads();
    
    if(index < boardInfoSetSize)
    {
        uint8_t oppoBoard[58];

        for(uint8_t byteInd=0; byteInd<58; byteInd++)
        {
            oppoBoard[byteInd] = oppoBoardInfoSet[index*58+byteInd];
        }

        char oneFenString[100];
        char* strFenIndex = oneFenString;
        
        // Print pieces
        for(int8_t row=7; row>=0; row--)
        {
            char rowString[8]; //[a-h]
            for(uint8_t col=0; col<8; col++)
            {
                uint8_t squareByteInd = row;
                uint8_t squareBitInd = col;
                char squareItem = 1;
                for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
                {
                    uint8_t pieceByteOffset = pieceInd*8;
                    
                    uint8_t* selfPieceTypeStart = selfBoard + pieceByteOffset;
                    uint8_t* selfPieceTypeRowByte = selfPieceTypeStart + squareByteInd;
                    bool selfPieceOnSquare = RBCAgetBit(*selfPieceTypeRowByte,8-1-squareBitInd);
                    squareItem = selfPieceOnSquare*((squareItem!=1)?'X':selfPiecesChar[pieceInd]) +
                                !selfPieceOnSquare*squareItem;
                    
                    uint8_t* oppoPieceTypeStart = oppoBoard + pieceByteOffset;
                    uint8_t* oppoPieceTypeRowByte = oppoPieceTypeStart + squareByteInd;
                    bool oppoPieceOnSquare = RBCAgetBit(*oppoPieceTypeRowByte,8-1-squareBitInd);
                    squareItem = oppoPieceOnSquare*((squareItem!=1)?'X':oppoPiecesChar[pieceInd]) +
                                !oppoPieceOnSquare*squareItem;
                }
                rowString[col] = squareItem;
            }
            uint8_t space = 0;
            for(uint8_t col=0; col<8; col++)
            {
                if(rowString[col]!=1)
                {
                    if(space!=0)
                    {
                        *strFenIndex = space+48;
                        strFenIndex++;
                    }
                    *strFenIndex = rowString[col];
                    strFenIndex++;
                }
                space = (rowString[col]==1)*(space+1) + (rowString[col]!=1)*0;
            }
            if(space!=0)
            {
                *strFenIndex = space+48;
                strFenIndex++;
            }
            if(row>0)
            {
                *strFenIndex = '/';
                strFenIndex++;
            }
        }
        
        *strFenIndex = ' ';
        strFenIndex++;
        
        //Print turn color
        *strFenIndex = turnColor;
        strFenIndex++;
        
        *strFenIndex = ' ';
        strFenIndex++;
        char* strFenIndexPreCastling = strFenIndex;
        
        //Print castling rights
        uint8_t* castlingWhite = (selfColor=='w')?(selfBoard+48):(oppoBoard+48);        
        bool castlingWhiteKingside = RBCAgetBit(*castlingWhite,7);
        *strFenIndex = 'K';
        if(castlingWhiteKingside)
            strFenIndex++;
        bool castlingWhiteQueenside = RBCAgetBit(*castlingWhite,6);
        *strFenIndex = 'Q';
        if(castlingWhiteQueenside)
            strFenIndex++;
        
        uint8_t* castlingBlack = (selfColor=='w')?(oppoBoard+48):(selfBoard+48);
        bool castlingBlackKingside = RBCAgetBit(*castlingBlack,7);
        *strFenIndex = 'k';
        if(castlingBlackKingside)
            strFenIndex++;
        bool castlingBlackQueenside = RBCAgetBit(*castlingBlack,6);
        *strFenIndex = 'q';
        if(castlingBlackQueenside)
            strFenIndex++;
        
        if(strFenIndex==strFenIndexPreCastling)
        {
            *strFenIndex = '-';
            strFenIndex++;
        }
        
        *strFenIndex = ' ';
        strFenIndex++;
        
        //En passant        
        uint8_t* enPassant = (turnColor==selfColor)?(oppoBoard+48):(selfBoard+48);
        uint8_t offset = 2;
        char enPas[2] = {'-',' '};
        for(int8_t row=7; row>=0; row--)
        {
            for(uint8_t col=0; col<8; col++)
            {
                uint8_t totalBitInd = (row*8)+col;
                uint8_t byte_offset = (totalBitInd+offset)/8;
                uint8_t bit_Ind = (totalBitInd+offset)%8;
                uint8_t readByte = *(enPassant+byte_offset);
                bool validEnPassant = RBCAgetBit(readByte,7-bit_Ind);
                bool doubleMatch = (validEnPassant && enPas[0]!='-');
                enPas[0] = doubleMatch*'X' + !doubleMatch*enPas[0];
                enPas[0] = validEnPassant*(col+'a') + !validEnPassant*enPas[0];
                enPas[1] = validEnPassant*(row+'1') + !validEnPassant*enPas[1];
            }
        }
        *strFenIndex = enPas[0];
        strFenIndex++;
        *strFenIndex = enPas[1];
        if(enPas[1]!=' ')
            strFenIndex++;
        *strFenIndex = ' ';
        strFenIndex++;

        //Halfmove number
        uint8_t* selfHalfMoveNumPtr = selfBoard+56;
        uint8_t selfHalfMoveNum = *selfHalfMoveNumPtr;
        selfHalfMoveNum = selfHalfMoveNum << 1;
        selfHalfMoveNum &= 127;
        bool selfHalfMoveNumLastBit = ((*(selfHalfMoveNumPtr+1))&128);
        selfHalfMoveNum = selfHalfMoveNumLastBit*(selfHalfMoveNum+1) + !selfHalfMoveNumLastBit*selfHalfMoveNum;
        
        uint8_t* oppoHalfMoveNumPtr = oppoBoard+56;
        uint8_t oppoHalfMoveNum = *oppoHalfMoveNumPtr;
        oppoHalfMoveNum = oppoHalfMoveNum << 1;
        oppoHalfMoveNum &= 127;
        bool oppoHalfMoveNumLastBit = ((*(oppoHalfMoveNumPtr+1))&128);
        oppoHalfMoveNum = oppoHalfMoveNumLastBit*(oppoHalfMoveNum+1) + !oppoHalfMoveNumLastBit*oppoHalfMoveNum;
       
        uint8_t halfMoveNum = (selfHalfMoveNum<oppoHalfMoveNum)?selfHalfMoveNum:oppoHalfMoveNum;
        numberToChar(&strFenIndex,halfMoveNum);
        
        *strFenIndex = ' ';
        strFenIndex++;
        
        numberToChar(&strFenIndex,nextMoveNumber);
        
        *strFenIndex = '\0';
        strFenIndex++;        
        
        char* thisFen = fenVector+(index*100);
        for(uint8_t i=0; i<100; i++)
        {
            *(thisFen+i) = oneFenString[i];
        }
    }
}

void RBCAgent::FullChessInfo::getAllFEN_GPU
(
    CIS::OnePlayerChessInfo& self,
    const PieceColor selfColor,
    std::unique_ptr<ChessInformationSet>& cis,
    const PieceColor nextTurn,
    const unsigned int nextCompleteTurn,
    std::vector<std::string>& allFEN
)
{    
    std::uint64_t cis_size = cis->size();
    std::uint64_t cis_byte_size = cis_size*(chessInfoSize/8);
    std::uint8_t* hostInfoSetPtr = cis->getInfoSetPtr();
    uint8_t* deviceOppoInfoSetPtr;
    CHECK(cudaMalloc((void**)&deviceOppoInfoSetPtr,cis_byte_size*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceOppoInfoSetPtr,hostInfoSetPtr,cis_byte_size*sizeof(uint8_t),cudaMemcpyHostToDevice));
        
    std::vector<char> whitePieces = {'P','N','B','R','Q','K'};
    std::vector<char> blackPieces = {'p','n','b','r','q','k'};
    
    char* deviceSelfPieceCharSet;
    CHECK(cudaMalloc((void**)&deviceSelfPieceCharSet,6*sizeof(char)));
    char* deviceOppoPieceCharSet;
    CHECK(cudaMalloc((void**)&deviceOppoPieceCharSet,6*sizeof(char)));
    char selfColorChar;
    if(selfColor==PieceColor::white)
    {
        CHECK(cudaMemcpy(deviceSelfPieceCharSet,whitePieces.data(),6*sizeof(char),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(deviceOppoPieceCharSet,blackPieces.data(),6*sizeof(char),cudaMemcpyHostToDevice));
        selfColorChar = 'w';
    }
    else
    {
        CHECK(cudaMemcpy(deviceSelfPieceCharSet,blackPieces.data(),6*sizeof(char),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(deviceOppoPieceCharSet,whitePieces.data(),6*sizeof(char),cudaMemcpyHostToDevice));
        selfColorChar = 'b';
    }
    
    char turnColor = (nextTurn==PieceColor::white)?'w':'b';
    
    std::array<std::uint8_t,58> selfBoardData;
    std::unique_ptr<std::bitset<chessInfoSize>> encodedBoard = cis->encodeBoard(self,1);
    for(int i=0; i<chessInfoSize; i+=8)
    {
        std::uint8_t byteInd = i/8;
        std::bitset<8> byteSet;
        for(int k=0;k<8;k++)
            byteSet[k] = (*encodedBoard)[i+k];
        std::uint8_t value = 0;
        for(int k=0;k<8;k++)
            if(byteSet[k])
                value |= 1<<(7-k);
        selfBoardData[byteInd] = value; 
    }
    uint8_t* deviceSelfInfoSetPtr;
    CHECK(cudaMalloc((void**)&deviceSelfInfoSetPtr,58*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceSelfInfoSetPtr,selfBoardData.data(),58*sizeof(uint8_t),cudaMemcpyHostToDevice));
    
    char* deviceFenVector;
    std::vector<char> hostFenVector(cis_size*100);
    CHECK(cudaMalloc((void**)&deviceFenVector,cis_size*100*sizeof(char)));
            
    int suggested_blockSize; 
    int suggested_minGridSize;
    cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, genFEN, 0, 0);
    int device;
    cudaGetDevice(&device); 
    struct cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);    
    
    dim3 blocks(suggested_blockSize);
    dim3 grids(ceil((float)cis_size/suggested_blockSize));
    
    genFEN<<<grids,blocks>>>
    (
        deviceOppoInfoSetPtr,
        deviceOppoPieceCharSet, // pawn=0,knight=1,bishop=2,rook=3,queen=4,king=5
        selfColorChar, // w or b
        deviceSelfInfoSetPtr,
        deviceSelfPieceCharSet, // pawn=0,knight=1,bishop=2,rook=3,queen=4,king=5
        turnColor, // w or b
        cis_size,
        nextCompleteTurn,
        deviceFenVector
    );
    cudaDeviceSynchronize();
        
    CHECK(cudaMemcpy(hostFenVector.data(),deviceFenVector,cis_size*100*sizeof(char),cudaMemcpyDeviceToHost));
    for(uint index=0; index<hostFenVector.size(); index+=100)
    {
        allFEN.push_back(std::string(hostFenVector.data()+index));
    }
    
    cudaFree(deviceOppoInfoSetPtr);
    cudaFree(deviceSelfPieceCharSet);
    cudaFree(deviceSelfInfoSetPtr);
    cudaFree(deviceOppoPieceCharSet);
    cudaFree(deviceFenVector);    
}
}
