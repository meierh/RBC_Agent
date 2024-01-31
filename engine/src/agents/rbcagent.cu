#include "rbcagent.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace crazyara {

__device__ bool getBit
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
    
    uint8_t* selfBoardInfoSet, // 6*64 pieces + 2*1 castling + 64 en_passant + 7 halfmove + 7 prob (in bits)
    char selfPieceCharSet[6], // pawn=0,knight=1,bishop=2,rook=3,queen=4,king=5
    char selfColor, // w or b
    uint8_t* opponentBoardInfoSet, // 6*64 pieces + 2*1 castling + 64 en_passant + 7 halfmove + 7 prob (in bits)
    char opponentPieceCharSet[6], // pawn=0,knight=1,bishop=2,rook=3,queen=4,king=5
    char turnColor, // w or b
    uint64_t boardInfoSetSize,
    uint16_t nextMoveNumber,
    char* fenVector // 100 byte per Item    
)
{   
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < boardInfoSetSize)
    {
        __shared__ uint8_t selfBoard[58];
        __shared__ char selfPiecesChar[6];
        uint8_t oppoBoard[58];
        __shared__ char oppoPiecesChar[6];
        for(uint8_t byteInd=0; byteInd<58; byteInd++)
        {
            selfBoard[byteInd] = selfBoardInfoSet[byteInd];
            oppoBoard[byteInd] = opponentBoardInfoSet[index*58+byteInd];
        }
        for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
        {
            selfPiecesChar[pieceInd] = selfPieceCharSet[pieceInd];
            oppoPiecesChar[pieceInd] = opponentPieceCharSet[pieceInd];
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
                    bool selfPieceOnSquare = getBit(*selfPieceTypeRowByte,8-1-squareBitInd);
                    squareItem = selfPieceOnSquare*((squareItem!=1)?'X':selfPiecesChar[pieceInd]) +
                                !selfPieceOnSquare*squareItem;
                    
                    uint8_t* oppoPieceTypeStart = oppoBoard + pieceByteOffset;
                    uint8_t* oppoPieceTypeRowByte = oppoPieceTypeStart + squareByteInd;
                    bool oppoPieceOnSquare = getBit(*oppoPieceTypeRowByte,8-1-squareBitInd);
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
        bool castlingWhiteKingside = getBit(*castlingWhite,7);
        *strFenIndex = 'K';
        if(castlingWhiteKingside)
            strFenIndex++;
        bool castlingWhiteQueenside = getBit(*castlingWhite,6);
        *strFenIndex = 'Q';
        if(castlingWhiteQueenside)
            strFenIndex++;
        
        uint8_t* castlingBlack = (selfColor=='w')?(oppoBoard+48):(selfBoard+48);
        bool castlingBlackKingside = getBit(*castlingBlack,7);
        *strFenIndex = 'k';
        if(castlingBlackKingside)
            strFenIndex++;
        bool castlingBlackQueenside = getBit(*castlingBlack,6);
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
        for(uint8_t row=0; row<8; row--)
        {
            for(uint8_t col=0; col<8; col++)
            {
                uint8_t totalBitInd = (row*8)+col;
                uint8_t byte_offset = (totalBitInd+offset)/8;
                uint8_t bit_Ind = (totalBitInd+offset)%8;
                uint8_t readByte = *(enPassant+byte_offset);
                bool validEnPassant = getBit(readByte,7-bit_Ind);
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
    const CIS::OnePlayerChessInfo& self,
    Player selfColor,
    std::unique_ptr<ChessInformationSet>& cis,
    const PieceColor nextTurn,
    const unsigned int nextCompleteTurn,
    std::vector<std::string> allFEN
)
{
    /*
    int suggested_blockSize; 
    int suggested_minGridSize;
    cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, genFEN, 0, 0);
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);    
    
    // block Size per dimension out of suggested_blockSize
    int provSqrt = (int) std::sqrt(suggested_blockSize);
    // make blockSize a multiplier of the device warp Size
    int warpMult = suggested_blockSize / deviceProp.warpSize;
   
	int block_dim_x;
    int block_dim_y;
    
    if(warpMult != 0)
    {
        block_dim_x = deviceProp.warpSize;
        block_dim_y = warpMult;
    }
    else
    {
        block_dim_x = provSqrt;
        block_dim_y = suggested_blockSize / provSqrt;
    }
    
	dim3 blocks(block_dim_x,block_dim_y);
    dim3 grids(ceil(input.cols/grids.x),ceil((float)input.rows/grids.y));    
    */

    /*
    std::cout<<"Mark boards that do not fit: ";
    for(auto clause : conditions)
        std::cout<<clause.to_string()<<"&&";
    std::cout<<std::endl;

    std::vector<std::vector<std::pair<std::array<std::uint8_t,48>,std::array<std::uint8_t,48>>>> hostBitwiseCondition;
    hostBitwiseCondition.resize(conditions.size());
    std::uint8_t numberOfConditions = conditions.size();
    std::vector<std::uint8_t> hostClausesPerCondition(numberOfConditions);
    if(numberOfConditions>25)
        throw std::logic_error("There must maximal 25 conditions");
    for(uint conditionInd=0; conditionInd<conditions.size(); conditionInd++)
    {
        conditions[conditionInd].to_bits(hostBitwiseCondition[conditionInd]);
        hostClausesPerCondition[conditionInd] = hostBitwiseCondition[conditionInd].size();
        if(hostBitwiseCondition[conditionInd].size()>2)
            throw std::logic_error("There must only be 2 clauses per condition");
    }
    
    uint8_t* deviceClausesPerCondition;
    CHECK(cudaMalloc((void**)&deviceClausesPerCondition,numberOfConditions*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceClausesPerCondition,hostClausesPerCondition.data(),
                         numberOfConditions*sizeof(uint8_t),cudaMemcpyHostToDevice));
    
    uint8_t* deviceBitwiseCondition[conditions.size()];
    for(uint conditionInd=0; conditionInd<conditions.size(); conditionInd++)
    {
        CHECK(cudaMalloc((void**)&(deviceBitwiseCondition[conditionInd]), 
                         hostClausesPerCondition[conditionInd]*sizeof(uint8_t)*96));
        CHECK(cudaMemcpy(deviceBitwiseCondition[conditionInd],hostBitwiseCondition[conditionInd].data(), 
                         numberOfConditions*sizeof(uint8_t)*96,cudaMemcpyHostToDevice));
    }
    
    std::uint64_t cis_size = size();
    std::uint64_t cis_byte_size = cis_size*(chessInfoSize/8);
    std::uint8_t* hostInfoSetPtr = getInfoSetPtr();
    uint8_t* deviceInfoSetPtr;
    CHECK(cudaMalloc((void**)&deviceInfoSetPtr,cis_byte_size*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceInfoSetPtr,hostInfoSetPtr,cis_byte_size*sizeof(uint8_t),cudaMemcpyHostToDevice));
    
    std::vector<std::uint8_t> hostIncompatibleBoards(cis_size);
    uint8_t* deviceIncompatibleBoards;
    CHECK(cudaMalloc((void**)&deviceIncompatibleBoards,cis_size*sizeof(uint8_t)));
    
    checkConditions<<<1,1>>>
    (
        numberOfConditions,
        deviceClausesPerCondition,
        deviceBitwiseCondition,
        deviceInfoSetPtr,
        cis_size,
        deviceIncompatibleBoards
    );
    
    CHECK(cudaMemcpy(deviceIncompatibleBoards,hostIncompatibleBoards.data(),
                     cis_size*sizeof(uint8_t),cudaMemcpyDeviceToHost));
    
    cudaFree(deviceClausesPerCondition);
    for(uint conditionInd=0; conditionInd<conditions.size(); conditionInd++)
    {
        cudaFree(deviceBitwiseCondition[conditionInd]);
    }
    cudaFree(deviceInfoSetPtr);
    cudaFree(deviceIncompatibleBoards);

    for(uint64_t boardIndex=0; boardIndex<hostIncompatibleBoards.size(); boardIndex++)
    {
        if(hostIncompatibleBoards[boardIndex]==1)
            incompatibleBoards.push(boardIndex);
    }
    */
}
}
