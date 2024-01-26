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

__global__ void checkConditions
(
    uint8_t numberOfConditions,
    uint8_t* clausesPerCondition,
    uint8_t** conditionArray,
    uint8_t* boardInfoSet,
    uint64_t boardInfoSetSize,
    uint8_t* incompatibleBoards    
)
{
    __shared__ uint8_t clauseNbr[25];
    __shared__ uint8_t clauses[25][2][2][48];
    for(uint condInd=0; condInd<numberOfConditions; condInd++)
    {
        uint8_t numberOfClauses = clausesPerCondition[condInd];
        clauseNbr[condInd] = numberOfClauses;        
        for(uint clauseInd=0; clauseInd<numberOfClauses; clauseInd++)
        {
            for(uint byteInd=0; byteInd<48; byteInd++)
            {
                clauses[condInd][clauseInd][0][byteInd] = conditionArray[condInd][clauseInd*116+byteInd];
                clauses[condInd][clauseInd][1][byteInd] = conditionArray[condInd][clauseInd*116+48+byteInd];
            }
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
}
}
