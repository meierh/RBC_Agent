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
    {
        sum += *(literal+byteInd) & *(board+byteInd);
        //printf("lit:%d & board:%d = %d\n",*(literal+byteInd),*(board+byteInd),*(literal+byteInd) & *(board+byteInd));
    }
    return (sum!=0)?true:false;
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
    __shared__ uint8_t clauses[25][2][49];
    uint8_t* clauseStart = conditionArray;
    if(threadIdx.x==0)
    {
        for(uint condInd=0; condInd<numberOfConditions; condInd++)
        {
            uint8_t numberOfClauses = clausesPerCondition[condInd];
            clauseNbr[condInd] = numberOfClauses;
            for(uint clauseInd=0; clauseInd<numberOfClauses; clauseInd++)
            {
                for(uint byteInd=0; byteInd<49; byteInd++)
                {
                    clauses[condInd][clauseInd][byteInd] = *(clauseStart+byteInd);
                }
                clauseStart += 49;
            }
        }
    }
    __syncthreads();
    
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
            //printf("oneClauseTrue %d\n",oneClauseTrue);
            // IMPORTANT: No clause can be be empty
            for(uint8_t clauseInd=0; clauseInd<thisCondClauseNbr; clauseInd++)
            {
                bool clauseDemandBool = (bool)clauses[condInd][clauseInd][0];
                uint8_t* clauseBits = clauses[condInd][clauseInd]+1;
                bool literalBool = evaluateLiteral(clauseBits,board);
                oneClauseTrue |= (clauseDemandBool==literalBool);
                //printf("oneClauseTrue %d\n",oneClauseTrue);
            }
            compatible &= oneClauseTrue;
        }
        //printf("compatible %d\n",compatible);
        incompatibleBoards[index] = (compatible)?1:0;
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
    
    std::unique_ptr<std::vector<std::uint8_t>> incompatibleBoard = checkBoardsValidGPU(conditions);
    // This is wrong
    std::for_each(incompatibleBoard->begin(),incompatibleBoard->end(),
                  [&](std::uint8_t boardIndex){incompatibleBoards.push(boardIndex);});
}

std::unique_ptr<std::vector<std::uint8_t>> ChessInformationSet::checkBoardsValidGPU
(
    const std::vector<BoardClause>& conditions
)
{
    /*
    std::cout<<"Mark boards that do not fit: ";
    for(auto clause : conditions)
        std::cout<<clause.to_string()<<"&&";
    std::cout<<std::endl;
    */
    
    std::vector<std::vector<std::pair<std::uint8_t,std::array<std::uint8_t,48>>>> hostBitwiseCondition;
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
    
    //std::cout<<"numberOfConditions:"<<int(numberOfConditions)<<std::endl;
    
    uint8_t* deviceClausesPerCondition;
    CHECK(cudaMalloc((void**)&deviceClausesPerCondition,numberOfConditions*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceClausesPerCondition,hostClausesPerCondition.data(),
                         numberOfConditions*sizeof(uint8_t),cudaMemcpyHostToDevice));
    
    //std::cout<<"Start remapping"<<std::endl;
    std::vector<std::uint8_t> remapClausesPerCondition(numberOfConditions);
    CHECK(cudaMemcpy(remapClausesPerCondition.data(),deviceClausesPerCondition,
                         numberOfConditions*sizeof(uint8_t),cudaMemcpyDeviceToHost));
    //std::for_each(remapClausesPerCondition.begin(),remapClausesPerCondition.end(),[](auto ind){std::cout<<int(ind)<<" ";});
    //std::cout<<"Remapped"<<std::endl;
    
    std::vector<std::pair<std::uint8_t,std::array<std::uint8_t,48>>> hostBitwiseConditionFlat;
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
    uint byteSizeMem = hostBitwiseConditionFlat.size()*sizeof(uint8_t)*49;
    CHECK(cudaMalloc((void**)&(deviceBitwiseCondition),byteSizeMem));
    CHECK(cudaMemcpy(deviceBitwiseCondition,hostBitwiseConditionFlat.data(),byteSizeMem,cudaMemcpyHostToDevice));    

    //std::cout<<"Start remapping"<<std::endl;
    std::vector<std::pair<std::uint8_t,std::array<std::uint8_t,48>>> remapHostBitwiseCondition(hostBitwiseConditionFlat.size());
    CHECK(cudaMemcpy(remapHostBitwiseCondition.data(),deviceBitwiseCondition,byteSizeMem,cudaMemcpyDeviceToHost));    
    
    /*
    for(auto oneCondition : remapHostBitwiseCondition)
    {
        std::cout<<"Cond:"<<std::endl;
        auto firstPart = oneCondition.first;
        std::cout<<" Must be: ";
        for(auto byte : firstPart)
            std::cout<<uint(byte)<<" ";
        std::cout<<std::endl;
        auto secondPart = oneCondition.second;
        std::cout<<" Must not be: ";
        for(auto byte : secondPart)
            std::cout<<uint(byte)<<" ";
        std::cout<<std::endl;
    }
    std::cout<<"Remapped"<<std::endl;
    */
    

    
    std::uint64_t cis_size = size();
    std::uint64_t cis_byte_size = cis_size*(chessInfoSize/8);
    std::uint8_t* hostInfoSetPtr = getInfoSetPtr();
    uint8_t* deviceInfoSetPtr;
    CHECK(cudaMalloc((void**)&deviceInfoSetPtr,cis_byte_size*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceInfoSetPtr,hostInfoSetPtr,cis_byte_size*sizeof(uint8_t),cudaMemcpyHostToDevice));
    
    //std::cout<<"cis_size:"<<int(cis_size)<<std::endl;
    //std::cout<<"cis_byte_size:"<<int(cis_byte_size)<<std::endl;
    
    auto result = std::make_unique<std::vector<std::uint8_t>>(cis_size);
    std::vector<std::uint8_t>& hostIncompatibleBoards = *result;
    uint8_t* deviceIncompatibleBoards;
    CHECK(cudaMalloc((void**)&deviceIncompatibleBoards,cis_size*sizeof(uint8_t)));
    
    //std::cout<<"Before kernel invocation"<<std::endl;
    
    int suggested_blockSize; 
    int suggested_minGridSize;
    cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, checkConditions, 0, 0);
    int device;
    cudaGetDevice(&device); 
    struct cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);    
    
    //std::cout<<"suggested_blockSize:"<<int(suggested_blockSize)<<std::endl;
    //std::cout<<"suggested_minGridSize:"<<int(suggested_minGridSize)<<std::endl;
    //std::cout<<"device:"<<int(device)<<std::endl;

    dim3 blocks(suggested_blockSize);
    //std::cout<<"blocks.x:"<<blocks.x<<std::endl;
    dim3 grids(ceil((float)cis_size/suggested_blockSize));
    //std::cout<<"grids.x:"<<grids.x<<std::endl;
    
    checkConditions<<<grids,blocks>>>
    (
        numberOfConditions,
        deviceClausesPerCondition,
        deviceBitwiseCondition,
        deviceInfoSetPtr,
        cis_size,
        deviceIncompatibleBoards
    );
    
    //std::cout<<"After kernel invocation"<<std::endl;
    
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
    
    //std::cout<<"End of function"<<std::endl;

    return result;
}

__global__ void initialReduce // blockDim.x == 32
(
    uint8_t* boardInfoSet,
    uint64_t boardInfoSetSize,
    uint32_t* boardSum //gridsize * (64*6)
)
{
    __shared__ uint32_t distro[32][6][64];
    for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
    {
        for(uint8_t squareInd=0; squareInd<64; squareInd++)
        {
            distro[threadIdx.x][pieceInd][squareInd] = 0;
        }
    }
    __syncthreads();
    uint64_t boardSize = 58;
    uint64_t blockSpan = ceilf((float)boardInfoSetSize / gridDim.x);
    uint64_t blockOffset = blockIdx.x*blockSpan;
    
    // Reduce to 32 boards shared memory
    uint8_t* blockStartPtr = boardInfoSet+blockOffset*boardSize;
    uint64_t validBlockSpan = min(blockSpan,boardInfoSetSize-blockOffset);
    for(uint64_t locIndex=threadIdx.x; locIndex<validBlockSpan; locIndex+=blockDim.x)
    {
        uint8_t* boardStart = blockStartPtr + locIndex*boardSize;
        uint8_t* probabilityPtr = boardStart+57;
        for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
        {
            uint8_t* pieceBoardStart = boardStart + pieceInd*8;
            uint8_t adding = 1; //*probabilityPtr & 127; // unset first bit
            for(uint8_t row=0; row<8; row++)
            {
                uint8_t* boardRow = pieceBoardStart + row;
                for(uint8_t col=0; col<8; col++)
                {
                    bool squareOccupied = CISgetBit(*boardRow,7-col);
                    distro[threadIdx.x][pieceInd][row*8+col] += (squareOccupied)?adding:0; 
                }
            }
        }
    }
    __syncthreads();
    
    for(uint8_t size=1; size<32; size++)
    {
        for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
        {
            distro[0][pieceInd][threadIdx.x] += distro[size][pieceInd][threadIdx.x];
            distro[0][pieceInd][threadIdx.x+32] += distro[size][pieceInd][threadIdx.x+32];
        }
    }
    __syncthreads();
    
    uint32_t* boardSumOffset = boardSum + blockIdx.x*(64*6);
    for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
    {
        *(boardSumOffset + (pieceInd*64) + threadIdx.x) = distro[0][pieceInd][threadIdx.x];
        *(boardSumOffset + (pieceInd*64) + threadIdx.x+32) = distro[0][pieceInd][threadIdx.x+32];
    }
}

__global__ void reduce // blockDim.x == 32
(
    uint32_t* boardsIn,
    uint32_t boardInSize,
    uint32_t* boardsOut //gridsize * (64*6)
)
{
    __shared__ uint32_t distro[32][6][64];
    for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
    {
        for(uint8_t squareInd=0; squareInd<64; squareInd++)
        {
            distro[threadIdx.x][pieceInd][squareInd] = 0;
        }
    }
    uint64_t blockSpan = ceilf((float)boardInSize / gridDim.x);
    uint64_t blockOffset = blockIdx.x*blockSpan;
    uint64_t blockOffsetData = blockOffset*6*64;

    for(uint64_t locIndex = 0; locIndex+threadIdx.x<blockSpan; locIndex+=blockDim.x)
    {
        uint32_t* boardStart = boardsIn + blockOffsetData + (locIndex+threadIdx.x)*6*64;
        for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
        {
            for(uint8_t squareInd=0; squareInd<64; squareInd++)
            {
                distro[threadIdx.x][pieceInd][squareInd] += *(boardStart+pieceInd*64+squareInd);
            }
        }
    }
    
    __syncthreads();
    for(uint8_t size=1; size<32; size++)
    {
        for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
        {
            distro[0][pieceInd][threadIdx.x] += distro[size][pieceInd][threadIdx.x];
            distro[0][pieceInd][threadIdx.x+32] += distro[size][pieceInd][threadIdx.x+32];
        }
    }
    
    __syncthreads();
    uint32_t* boardSumOffset = boardsOut + blockIdx.x*(64*6);
    for(uint8_t pieceInd=0; pieceInd<6; pieceInd++)
    {
        *(boardSumOffset + (pieceInd*64) + threadIdx.x) = distro[0][pieceInd][threadIdx.x];
        *(boardSumOffset + (pieceInd*64) + threadIdx.x+32) = distro[0][pieceInd][threadIdx.x+32];
    }
}

std::unique_ptr<ChessInformationSet::Distribution> ChessInformationSet::computeDistributionGPU()
{
    //std::cout<<"Compute Distribution"<<std::endl;
    std::uint64_t cis_size = size();
    std::uint64_t maxSize = 1;
    maxSize = maxSize<<32;
    if(cis_size >= maxSize)
    {
        std::cout<<"cis_size:"<<cis_size<<" > "<<maxSize<<std::endl;
        throw std::invalid_argument("CIS size too big");
    }
    std::uint64_t cis_byte_size = cis_size*(chessInfoSize/8);
    std::uint8_t* hostInfoSetPtr = getInfoSetPtr();
    uint8_t* deviceInfoSetPtr;
    CHECK(cudaMalloc((void**)&deviceInfoSetPtr,cis_byte_size*sizeof(uint8_t)));
    CHECK(cudaMemcpy(deviceInfoSetPtr,hostInfoSetPtr,cis_byte_size*sizeof(uint8_t),cudaMemcpyHostToDevice));
    
    /*
    for(int i=0; i<cis_size; i++)
    {
        for(int k=0; k<58; k++)
            std::cout<<int(hostInfoSetPtr[i*58+k])<<" ";
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    */
        
    //std::cout<<"cis_size:"<<int(cis_size)<<std::endl;
    //std::cout<<"cis_byte_size:"<<int(cis_byte_size)<<std::endl;
    
    dim3 blocks(32);
    
    std::uint64_t maxNbrBlocks = 60000;
    std::uint64_t boardsPerBlock = cis_size / maxNbrBlocks;
    std::uint64_t boardsPerThread = boardsPerBlock / blocks.x;
    std::uint64_t minBoardsPerThread = 8;
    if(boardsPerThread < minBoardsPerThread)
        boardsPerThread = minBoardsPerThread;    
    dim3 grids(ceil((float)cis_size/(blocks.x*boardsPerThread)));

    //std::cout<<"blocks.x:"<<blocks.x<<std::endl;
    //std::cout<<"grids.x:"<<grids.x<<std::endl;
    
    uint32_t* deviceBoardSumIn;
    CHECK(cudaMalloc((void**)&deviceBoardSumIn,grids.x*6*64*sizeof(uint32_t)));
    
    uint32_t* deviceBoardSumOut;
    CHECK(cudaMalloc((void**)&deviceBoardSumOut,grids.x*6*64*sizeof(uint32_t)));
    
    initialReduce<<<grids,blocks>>>
    (
        deviceInfoSetPtr,
        cis_size,
        deviceBoardSumIn
    );
    cudaDeviceSynchronize();
    //std::cout<<"Initial reduction"<<std::endl;
    
    std::uint64_t inGridSize = grids.x;
    std::uint64_t outGridSize;
    while(inGridSize>1)
    {
        outGridSize = ceil((float)inGridSize / 128);
        grids = dim3(outGridSize);
        //std::cout<<"inGridSize:"<<inGridSize<<std::endl;
        //std::cout<<"outGridSize:"<<outGridSize<<std::endl;
        reduce<<<grids,blocks>>>
        (
            deviceBoardSumIn,
            inGridSize,
            deviceBoardSumOut
        );
        cudaDeviceSynchronize();
        inGridSize = outGridSize;
        uint32_t* temp = deviceBoardSumIn;
        deviceBoardSumIn = deviceBoardSumOut;
        deviceBoardSumOut = temp;
    }
    //std::cout<<"Copy result back"<<std::endl;
    std::array<std::uint32_t,384> piecesSum;
    CHECK(cudaMemcpy(piecesSum.data(),deviceBoardSumIn,384*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    
    /*
    for(auto count : piecesSum)
        std::cout<<count<<" ";
    std::cout<<std::endl;
    */
    
    //std::cout<<"Free"<<std::endl;
    cudaFree(deviceInfoSetPtr);
    cudaFree(deviceBoardSumIn);
    cudaFree(deviceBoardSumOut);
    
    //std::cout<<"Compute Fraction"<<std::endl;
    std::array<double,384> piecesSumDouble;
    for(uint i=0; i<piecesSum.size(); i++)
        piecesSumDouble[i] = static_cast<double>(piecesSum[i]) / cis_size;
    
    //std::cout<<"Compute Distribution"<<std::endl;
    auto piecesDistro = std::make_unique<Distribution>();
    std::memcpy(piecesDistro->pawns.data(),  piecesSumDouble.data(),    64*sizeof(double));
    std::memcpy(piecesDistro->knights.data(),piecesSumDouble.data()+64, 64*sizeof(double));
    std::memcpy(piecesDistro->bishops.data(),piecesSumDouble.data()+128,64*sizeof(double));
    std::memcpy(piecesDistro->rooks.data(),  piecesSumDouble.data()+192,64*sizeof(double));
    std::memcpy(piecesDistro->queens.data(), piecesSumDouble.data()+256,64*sizeof(double));
    std::memcpy(piecesDistro->kings.data(),  piecesSumDouble.data()+320,64*sizeof(double));
    //std::cout<<"Return"<<std::endl;
    return piecesDistro;
}
}
