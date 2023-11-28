#include <iostream>
#include <random>
#include <bitset>
#include <unordered_set>
#include "informationset.h"

namespace crazyara {
template< size_t size>
typename std::bitset<size> random_bitset( double p = 0.5)
{
    typename std::bitset<size> bits;
    std::random_device rd;
    std::mt19937 gen( rd());
    std::bernoulli_distribution d( p);

    for( int n = 0; n < size; ++n)
    {
        bits[ n] = d( gen);
    }

    return bits;
}

template<size_t size>
std::bitset<size> gen_random_bitset(double p=0.5)
{
    return random_bitset<size>(p);
}

template<size_t size>
struct BitsetGenerator {
    double p;
    BitsetGenerator (double p) : p(p) {}
    std::bitset<size> operator() () { return gen_random_bitset<size>(p); }
};

constexpr uint bitSize = 27;

TEST(informationset_test, constructor_test)
{
    InformationSet<bitSize> ifs;    
    EXPECT_EQ(ifs.size(), 0);
    ifs = InformationSet<bitSize>(0);
    EXPECT_EQ(ifs.size(), 0);
    ifs = InformationSet<bitSize>(5);
    EXPECT_EQ(ifs.size(), 0);
}

TEST(informationset_test, setBit_test)
{   
    InformationSet<bitSize> ifs;
    std::uint64_t var = 0;
    EXPECT_EQ(var,0);
    ifs.setBit<std::uint64_t>(var,0);
    EXPECT_EQ(var,1);
    ifs.setBit<std::uint64_t>(var,3);
    EXPECT_EQ(var,9);
    ifs.setBit<std::uint64_t>(var,8);
    EXPECT_EQ(var,265);
    ifs.setBit<std::uint64_t>(var,30);
    EXPECT_EQ(var,1073742089);
    ifs.setBit<std::uint64_t>(var,15);
    EXPECT_EQ(var,1073774857);
    ifs.setBit<std::uint64_t>(var,8);
    EXPECT_EQ(var,1073774857);
}

TEST(informationset_test, unsetBit_test)
{   
    InformationSet<bitSize> ifs;
    std::uint64_t var = 1073774857;
    EXPECT_EQ(var,1073774857);
    ifs.unsetBit<std::uint64_t>(var,0);
    EXPECT_EQ(var,1073774856);
    ifs.unsetBit<std::uint64_t>(var,3);
    EXPECT_EQ(var,1073774848);
    ifs.unsetBit<std::uint64_t>(var,8);
    EXPECT_EQ(var,1073774592);
    ifs.unsetBit<std::uint64_t>(var,30);
    EXPECT_EQ(var,32768);
    ifs.unsetBit<std::uint64_t>(var,15);
    EXPECT_EQ(var,0);
    ifs.unsetBit<std::uint64_t>(var,8);
    EXPECT_EQ(var,0);
}

TEST(informationset_test, getBit_test)
{
    InformationSet<bitSize> ifs;
    std::uint8_t var0 = 13;
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var0,0),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var0,1),false);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var0,2),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var0,3),true);
    
    std::uint64_t var = 0x596D;
    EXPECT_EQ(var,0x596D);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,0),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,1),false);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,2),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,3),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,4),false);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,5),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,6),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,7),false);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,8),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,9),false);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,10),false);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,11),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,12),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,13),false);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,14),true);
    EXPECT_EQ(ifs.getBit<std::uint64_t>(var,15),false);
}

TEST(informationset_test, assignBitPattern_test)
{
    InformationSet<bitSize> ifs;   
    std::uint64_t var = 0x00F;
    std::bitset<bitSize> bits;
    ifs.assignBitPattern<std::uint64_t>(bits,0,var,4);
    EXPECT_EQ(bits,std::bitset<bitSize>({0x000000F}));
    
    var = 0x0FA;
    ifs.assignBitPattern<std::uint64_t>(bits,20,var,8);
    EXPECT_EQ(bits,std::bitset<bitSize>({0x5F0000F}));
    
    var = 0x1357;
    ifs.assignBitPattern<std::uint64_t>(bits,4,var,16);
    EXPECT_EQ(bits,std::bitset<bitSize>({0x5FEAC8F}));
    
    var = 0x0000;
    ifs.assignBitPattern<std::uint64_t>(bits,30,var,16);
    EXPECT_EQ(bits,std::bitset<bitSize>({0x5FEAC8F}));
}

TEST(informationset_test, transferBitPattern_test)
{
    InformationSet<27> ifs;   
    std::bitset<27> bits({0x5FEAC8F});
    EXPECT_EQ(ifs.transferBitPattern<std::uint64_t>(bits,0,4),0xF);
    EXPECT_EQ(ifs.transferBitPattern<std::uint64_t>(bits,4,4),0x1);    
    EXPECT_EQ(ifs.transferBitPattern<std::uint64_t>(bits,24,12),0x5);
}

TEST(informationset_test, transferBytePattern_test)
{
    InformationSet<bitSize> ifs;   
    std::bitset<8> bits1({0x59});
    EXPECT_EQ(ifs.transferBytePattern(bits1),0x9A);
    std::bitset<8> bits2({0xC1});
    EXPECT_EQ(ifs.transferBytePattern(bits2),0x83);    
}

TEST(informationset_test, writeSingleOnNonParam_test)
{
    crazyara::InformationSet<bitSize> ifs;
    std::vector<std::bitset<bitSize>> bitsets(32,random_bitset<bitSize>(0.5));
    for(std::bitset<bitSize>& bitset : bitsets)
        ifs.add(bitset);
    EXPECT_EQ(ifs.size(), 32);
    
    ifs.add(bitsets);
    EXPECT_EQ(ifs.size(), 64);
}

TEST(informationset_test, readWriteBits_test)
{   
    crazyara::InformationSet<bitSize> ifs;   
    std::vector<std::bitset<bitSize>> bitsets(128,random_bitset<bitSize>(0.5));
    ifs.add(bitsets);
    EXPECT_EQ(ifs.size(), 128);
    for(uint i=0; i<bitsets.size(); i++)
        EXPECT_EQ(*(ifs.getBitPattern(i)),bitsets[i]);
}

TEST(informationset_test, readWriteIteratorComp_test)
{   
    crazyara::InformationSet<bitSize> ifs;
    std::vector<std::bitset<bitSize>> bitsets(64,random_bitset<bitSize>(0.5));
    ifs.add(bitsets);
    auto iter = ifs.cbegin();
    for(int i=0; i<bitsets.size(); i++,iter++)
    {
        std::unique_ptr<std::bitset<bitSize>> data = *iter;
        EXPECT_EQ(*data,bitsets[i]);
        iter++;
    }
}

TEST(informationset_test, readWriteIndexComp_test)
{   
    crazyara::InformationSet<bitSize> ifs;
    std::vector<std::bitset<bitSize>> bitsets(1,random_bitset<bitSize>(0.5));
    ifs.add(bitsets);
    EXPECT_EQ(ifs.size(),bitsets.size());
    
    for(int i=0; i<bitsets.size(); i++)
    {
        std::unique_ptr<std::bitset<bitSize>> data = ifs.getBitPattern(i);
        EXPECT_EQ(*data,bitsets[i]);
    }
}

TEST(informationset_test, readWriteIterDeleteCompSameSize_test)
{   
    crazyara::InformationSet<bitSize> ifs;
    std::vector<std::bitset<bitSize>> bitsets(64);
    std::generate(bitsets.begin(),bitsets.end(),BitsetGenerator<bitSize>(0.5));
    ifs.add(bitsets);
    std::vector<std::uint64_t> prime = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61};
    std::vector<std::bitset<bitSize>> replace(prime.size());
    std::generate(replace.begin(),replace.end(),BitsetGenerator<bitSize>(0.5));
    ifs.remove(prime);
    ifs.add(replace);
    for(uint i=0; i<prime.size(); i++)
        bitsets[prime[i]] = replace[i];

    auto iter = ifs.cbegin();
    for(int i=0; i<bitsets.size(); i++,iter++)
    {
        std::unique_ptr<std::bitset<bitSize>> data = *iter;
        EXPECT_EQ(*data,bitsets[i]);
    }
}

TEST(informationset_test, readWriteIndexDeleteCompSameSize_test)
{   
    crazyara::InformationSet<bitSize> ifs;
    std::vector<std::bitset<bitSize>> bitsets(64);
    std::generate(bitsets.begin(),bitsets.end(),BitsetGenerator<bitSize>(0.5));
    ifs.add(bitsets);
    std::vector<std::uint64_t> prime = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61};
    std::vector<std::bitset<bitSize>> replace(prime.size());
    std::generate(replace.begin(),replace.end(),BitsetGenerator<bitSize>(0.5));
    ifs.remove(prime);
    ifs.add(replace);
    for(uint i=0; i<prime.size(); i++)
        bitsets[prime[i]] = replace[i];
        
    auto iter = ifs.cbegin();
    for(int i=0; i<bitsets.size(); i++,iter++)
    {
        std::unique_ptr<std::bitset<bitSize>> data = ifs.getBitPattern(i);
        EXPECT_EQ(*data,bitsets[i]);
        iter++;
    }
}

TEST(informationset_test, readWriteIterDeleteCompIncreaseSize_test)
{   
    crazyara::InformationSet<bitSize> ifs;
        
    std::vector<std::bitset<bitSize>> bitsets(64);
    std::generate(bitsets.begin(),bitsets.end(),BitsetGenerator<bitSize>(0.5));
    ifs.add(bitsets);
    std::vector<std::uint64_t> prime = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61};
    std::vector<std::bitset<bitSize>> replace(2*prime.size());
    std::generate(replace.begin(),replace.end(),BitsetGenerator<bitSize>(0.5));
    ifs.remove(prime);
    ifs.add(replace);
    
    for(uint i=0; i<prime.size(); i++)
        bitsets[prime[i]] = replace[i];
    for(uint i=prime.size(); i<replace.size(); i++)
        bitsets.push_back(replace[i]);
    std::unordered_multiset<std::bitset<bitSize>> bitsetSet(bitsets.begin(),bitsets.end());
    
    for(auto iter=ifs.cbegin(); iter!=ifs.cend(); iter++)
    {
        auto iterSet = bitsetSet.find(**iter);
        EXPECT_EQ(iterSet==bitsetSet.end(),false);
        bitsetSet.erase(iterSet);
    }
    EXPECT_EQ(bitsetSet.empty(),true);
}

TEST(informationset_test, readWriteIndexDeleteCompDecreaseSize_test)
{   
    crazyara::InformationSet<bitSize> ifs;
        
    std::vector<std::bitset<bitSize>> bitsets(64);
    std::generate(bitsets.begin(),bitsets.end(),BitsetGenerator<bitSize>(0.5));
    ifs.add(bitsets);
    std::vector<std::uint64_t> prime = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61};
    ifs.remove(prime);
    std::unordered_set<std::uint64_t> primeSet(prime.begin(),prime.end());
    
    std::vector<std::bitset<bitSize>> bitsetsErased;
    for(uint i=0; i<bitsets.size(); i++)
        if(primeSet.find(i)==primeSet.end())
            bitsetsErased.push_back(bitsets[i]);

    std::unordered_multiset<std::bitset<bitSize>> bitsetSet(bitsetsErased.begin(),bitsetsErased.end());
    for(auto iter=ifs.cbegin(); iter!=ifs.cend(); iter++)
    {
        auto iterSet = bitsetSet.find(**iter);
        EXPECT_EQ(iterSet==bitsetSet.end(),false);
        if(iterSet!=bitsetSet.end())
            bitsetSet.erase(iterSet);
    }
    EXPECT_EQ(bitsetSet.empty(),true);
}

TEST(informationset_test, deleteWithRemoveNoReset_test)
{   
    crazyara::InformationSet<bitSize> ifs;
        
    std::vector<std::bitset<bitSize>> bitsets(64);
    std::generate(bitsets.begin(),bitsets.end(),BitsetGenerator<bitSize>(0.5));
    ifs.add(bitsets);
    std::vector<std::uint64_t> prime = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61};
    ifs.remove(prime);
    ifs.clearRemoved(false);
    std::unordered_set<std::uint64_t> primeSet(prime.begin(),prime.end());
    
    std::vector<std::bitset<bitSize>> bitsetsErased;
    for(uint i=0; i<bitsets.size(); i++)
        if(primeSet.find(i)==primeSet.end())
            bitsetsErased.push_back(bitsets[i]);

    std::unordered_multiset<std::bitset<bitSize>> bitsetSet(bitsetsErased.begin(),bitsetsErased.end());
    for(auto iter=ifs.cbegin(); iter!=ifs.cend(); iter++)
    {
        auto iterSet = bitsetSet.find(**iter);
        EXPECT_EQ(iterSet==bitsetSet.end(),false);
        if(iterSet!=bitsetSet.end())
            bitsetSet.erase(iterSet);
    }
    EXPECT_EQ(bitsetSet.empty(),true);
}

TEST(informationset_test, deleteWithRemoveReset_test)
{   
    crazyara::InformationSet<bitSize> ifs;
        
    std::vector<std::bitset<bitSize>> bitsets(64);
    std::generate(bitsets.begin(),bitsets.end(),BitsetGenerator<bitSize>(0.5));
    ifs.add(bitsets);
    std::vector<std::uint64_t> prime = {2,3,5,7,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,41,43,47,53,54,55,56,57,58,59,61};
    ifs.remove(prime);
    ifs.clearRemoved(true);
    std::unordered_set<std::uint64_t> primeSet(prime.begin(),prime.end());
    
    std::vector<std::bitset<bitSize>> bitsetsErased;
    for(uint i=0; i<bitsets.size(); i++)
        if(primeSet.find(i)==primeSet.end())
            bitsetsErased.push_back(bitsets[i]);

    std::unordered_multiset<std::bitset<bitSize>> bitsetSet(bitsetsErased.begin(),bitsetsErased.end());
    for(auto iter=ifs.cbegin(); iter!=ifs.cend(); iter++)
    {
        auto iterSet = bitsetSet.find(**iter);
        EXPECT_EQ(iterSet==bitsetSet.end(),false);
        if(iterSet!=bitsetSet.end())
            bitsetSet.erase(iterSet);
    }
    EXPECT_EQ(bitsetSet.empty(),true);
}

};
