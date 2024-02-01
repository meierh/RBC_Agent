#include "chessinformationset.h"

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

__device__ bool evaluateLiteral
(
    uint8_t* literal,
    uint8_t* board
)
{
    uint8_t sum = 0;
    for(uint8_t byteInd=0; byteInd<48; byteInd++)
        sum += *(literal+byteInd) & *(board+byteInd);
    return (sum==0)?true:false;
}

__device__ bool CISgetBit
(
    uint8_t byte,
    uint8_t position
)
{
    return byte & (1 << position);
}

__global__ void checkConditions
(
    uint8_t numberOfConditions,
    uint8_t* clausesPerCondition,
    uint8_t* conditionArray,
    uint8_t* boardInfoSet,
    uint64_t boardInfoSetSize,
    uint8_t* incompatibleBoards    
)
{
    __shared__ uint8_t clauseNbr[25];
    __shared__ uint8_t clauses[25][2][2][48];
    uint8_t* clauseStart = conditionArray;
    for(uint condInd=0; condInd<numberOfConditions; condInd++)
    {
        uint8_t numberOfClauses = clausesPerCondition[condInd];
        clauseNbr[condInd] = numberOfClauses;
        for(uint clauseInd=0; clauseInd<numberOfClauses; clauseInd++)
        {
            for(uint byteInd=0; byteInd<48; byteInd++)
            {
                clauses[condInd][clauseInd][0][byteInd] = *(clauseStart+byteInd);
                clauses[condInd][clauseInd][1][byteInd] = *(clauseStart+48+byteInd);
            }
            clauseStart += 96;
        }
    }
    
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < boardInfoSetSize)
    {
        uint8_t board[58];
        for(uint8_t byteInd=0; byteInd<58; byteInd++)
        {
            board[byteInd] = boardInfoSet[index*58+byteInd];
        }
        
        bool compatible = true;
        for(uint8_t condInd=0; condInd<numberOfConditions; condInd++)
        {
            bool oneClauseTrue = false;
            uint8_t thisCondClauseNbr = clauseNbr[condInd];
            
            // IMPORTANT: No clause can be be empty
            for(uint8_t clauseInd=0; clauseInd<thisCondClauseNbr; clauseInd++)
            {
                uint8_t* demandPieceLiteral = clauses[condInd][clauseInd][0];
                oneClauseTrue |= evaluateLiteral(demandPieceLiteral,board);
                
                uint8_t* demandNonPieceLiteral = clauses[condInd][clauseInd][1];
                oneClauseTrue |= !evaluateLiteral(demandNonPieceLiteral,board);
            }
            compatible &= oneClauseTrue;
        }
        incompatibleBoards[index] = (compatible)?0:1;
    }
}

void ChessInformationSet::markIncompatibleBoardsGPU
(
    const std::vector<BoardClause>& conditions
)
{
    std::cout<<"Mark boards that do not fit: ";
    for(auto clause : conditions)
        std::cout<<clause.to_string()<<"&&";
    std::cout<<std::endl;
    
    std::unique_ptr<std::vector<std::uint8_t>> incompatibleBoard = getIncompatibleBoardsGPU(conditions);
    std::for_each(incompatibleBoard->begin(),incompatibleBoard->end(),
                  [&](std::uint8_t boardIndex){incompatibleBoards.push(boardIndex);});
}

std::unique_ptr<std::vector<std::uint8_t>> ChessInformationSet::getIncompatibleBoardsGPU
(
    const std::vector<BoardClause>& conditions
)
{
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
    for(uint conditionInd=0; conditionInd<numberOfConditions; conditionInd++)
    {
        conditions[conditionInd].to_bits(hostBitwiseCondition[conditionInd]);
        hostClausesPerCondition[conditionInd] = hostBitwiseCondition[conditionInd].size();
        if(hostBitwiseCondition[conditionInd].size()>2)
            throw std::logic_error("There must only be 2 clauses per condition");
    }
    
    std::cout<<"numberOfConditions:"<<int(numberOfConditions)<<std::endl;
    
    uint8_t* deviceClausesPerCondition;
    CHECK(cudaMalloc((void**)&deviceClausesPerCondition,numberOfConditions*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceClausesPerCondition,hostClausesPerCondition.data(),
                         numberOfConditions*sizeof(uint8_t),cudaMemcpyHostToDevice));
    
    std::cout<<"Start remapping"<<std::endl;
    std::vector<std::uint8_t> remapClausesPerCondition(numberOfConditions);
    CHECK(cudaMemcpy(remapClausesPerCondition.data(),deviceClausesPerCondition,
                         numberOfConditions*sizeof(uint8_t),cudaMemcpyDeviceToHost));
    std::for_each(remapClausesPerCondition.begin(),remapClausesPerCondition.end(),[](auto ind){std::cout<<int(ind)<<" ";});
    std::cout<<"Remapped"<<std::endl;
    
    std::vector<std::pair<std::array<std::uint8_t,48>,std::array<std::uint8_t,48>>> hostBitwiseConditionFlat;
    for(auto oneCondition : hostBitwiseCondition)
    {
        /*
        for(auto oneClause : oneCondition)
        {
            auto firstPart = oneClause.first;
            for(auto byte : firstPart)
                std::cout<<uint(byte)<<" ";
            std::cout<<std::endl;
            auto secondPart = oneClause.second;
            for(auto byte : secondPart)
                std::cout<<uint(byte)<<" ";
            std::cout<<std::endl;
        }
        */
        hostBitwiseConditionFlat.insert(hostBitwiseConditionFlat.end(),oneCondition.begin(),oneCondition.end());
    }
    
    uint8_t* deviceBitwiseCondition;
    uint byteSizeMem = hostBitwiseConditionFlat.size()*sizeof(uint8_t)*96;
    CHECK(cudaMalloc((void**)&(deviceBitwiseCondition),byteSizeMem));
    CHECK(cudaMemcpy(deviceBitwiseCondition,hostBitwiseConditionFlat.data(),byteSizeMem,cudaMemcpyHostToDevice));    

    std::cout<<"Start remapping"<<std::endl;
    std::vector<std::pair<std::array<std::uint8_t,48>,std::array<std::uint8_t,48>>> remapHostBitwiseCondition(hostBitwiseConditionFlat.size());
    CHECK(cudaMemcpy(remapHostBitwiseCondition.data(),deviceBitwiseCondition,byteSizeMem,cudaMemcpyDeviceToHost));    
    for(auto oneCondition : remapHostBitwiseCondition)
    {
        auto firstPart = oneCondition.first;
        for(auto byte : firstPart)
            std::cout<<uint(byte)<<" ";
        std::cout<<std::endl;
        auto secondPart = oneCondition.second;
        for(auto byte : secondPart)
            std::cout<<uint(byte)<<" ";
        std::cout<<std::endl;
    }
    std::cout<<"Remapped"<<std::endl;

    
    std::uint64_t cis_size = size();
    std::uint64_t cis_byte_size = cis_size*(chessInfoSize/8);
    std::uint8_t* hostInfoSetPtr = getInfoSetPtr();
    uint8_t* deviceInfoSetPtr;
    CHECK(cudaMalloc((void**)&deviceInfoSetPtr,cis_byte_size*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceInfoSetPtr,hostInfoSetPtr,cis_byte_size*sizeof(uint8_t),cudaMemcpyHostToDevice));
    
    std::cout<<"cis_size:"<<int(cis_size)<<std::endl;
    std::cout<<"cis_byte_size:"<<int(cis_byte_size)<<std::endl;
    
    auto result = std::make_unique<std::vector<std::uint8_t>>(cis_size);
    std::vector<std::uint8_t>& hostIncompatibleBoards = *result;
    uint8_t* deviceIncompatibleBoards;
    CHECK(cudaMalloc((void**)&deviceIncompatibleBoards,cis_size*sizeof(uint8_t)));
    
    std::cout<<"Before kernel invocation"<<std::endl;
    
    int suggested_blockSize; 
    int suggested_minGridSize;
    cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, checkConditions, 0, 0);
    int device;
    cudaGetDevice(&device); 
    struct cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);    
    
    std::cout<<"suggested_blockSize:"<<int(suggested_blockSize)<<std::endl;
    std::cout<<"suggested_minGridSize:"<<int(suggested_minGridSize)<<std::endl;
    std::cout<<"device:"<<int(device)<<std::endl;

    dim3 blocks(suggested_blockSize);
    std::cout<<"blocks.x:"<<blocks.x<<std::endl;
    dim3 grids(ceil((float)cis_size/suggested_blockSize));
    std::cout<<"grids.x:"<<grids.x<<std::endl;
    
    checkConditions<<<grids,blocks>>>
    (
        numberOfConditions,
        deviceClausesPerCondition,
        deviceBitwiseCondition, // Is still host pointer
        deviceInfoSetPtr,
        cis_size,
        deviceIncompatibleBoards
    );
    
    std::cout<<"After kernel invocation"<<std::endl;
    
    CHECK(cudaMemcpy(hostIncompatibleBoards.data(),deviceIncompatibleBoards,
                     cis_size*sizeof(uint8_t),cudaMemcpyDeviceToHost));
    
    cudaFree(deviceClausesPerCondition);
    cudaFree(deviceBitwiseCondition);
    cudaFree(deviceInfoSetPtr);
    cudaFree(deviceIncompatibleBoards);

    for(uint64_t boardIndex=0; boardIndex<hostIncompatibleBoards.size(); boardIndex++)
    {
        if(hostIncompatibleBoards[boardIndex]==1)
            incompatibleBoards.push(boardIndex);
    }
    
    std::cout<<"End of function"<<std::endl;

    return result;
}

__global__ void initialSum // max threads == 256
(
    uint8_t* boardInfoSet,
    uint64_t boardInfoSetSize,
    uint64_t* boardSum    
)
{
    __shared__ uint8_t boards[256][6][8];
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index >= boardInfoSetSize)
        return;
    
    for(uint pieceType=0; pieceType<6; pieceType++)
    {
        for(uint row=0; row<8; row++)
        {
            boards[threadIdx.x][pieceType][row] = boardInfoSet[index+pieceType*8+row];
        }
    }
    __syncthreads();
    
    for(uint8_t pieceType=0; pieceType<6; pieceType++)
    {
        uint64_t onePieceBoard[64];
        for(uint8_t row=0; row<8; row++)
        {
            for(uint8_t col=0; col<8; col++)
            {
                if(CISgetBit(boards[threadIdx.x][pieceType][row],7-col))
                    onePieceBoard[row*8+col]++;
            }
        }
        for(uint8_t squareInd=0; squareInd<64; squareInd++)
            *(boardSum+(blockIdx.x*6*64)+pieceType*64+squareInd) = onePieceBoard[squareInd];
    }
}

std::unique_ptr<ChessInformationSet::Distribution> ChessInformationSet::computeDistributionGPU()
{
    
}

}
