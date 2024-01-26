#include "rbcagent.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace crazyara {

void CHECK(cudaError_t cuError)
{
    if(cuError!=cudaSuccess)
    {   
        std::string cudaErrorString(cudaGetErrorString(cuError));
        std::cout<<"CUDA Error: "<<cudaErrorString<<std::endl;
        throw std::logic_error("CUDA Fail");
    }
}

__device__ bool getBit
(
    uint8_t byte,
    uint8_t position
)
{
    return byte & (1 << position);
}

__global__ void genFEN
(
    uint8_t* selfBoardInfoSet, // 6*64 pieces + 2*1 castling + 64 en_passant + 7 halfmove + 7 prob (in bits)
    uint8_t* opponentBoardInfoSet, // 6*64 pieces + 2*1 castling + 64 en_passant + 7 halfmove + 7 prob (in bits)
    uint64_t boardInfoSetSize,
    char* fenVector // 100 byte per Item    
)
{    
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < boardInfoSetSize)
    {
        uint8_t oppoBoard[58];
        uint8_t selfBoard[58];
        for(uint8_t byteInd=0; byteInd<58; byteInd++)
        {
            oppoBoard[byteInd] = opponentBoardInfoSet[index*58+byteInd];
            selfBoard[byteInd] = selfBoardInfoSet[byteInd];
        }
        uint64_t* oppoPieces = oppoBoard;
        uint64_t* selfPieces = selfBoard;
        
        
        char* thisFen = fenVector+(index*100);

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
