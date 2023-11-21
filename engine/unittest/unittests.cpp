#include <gtest/gtest.h>
TEST(sample_test_case, sample_test)
{
    EXPECT_EQ(1, 1);
}

#include <iostream>
#include <random>
#include <bitset>
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
    std::uint64_t var = 0x900;
    std::bitset<bitSize> bits;
    ifs.assignBitPattern<std::uint64_t>(bits,7,var,8);
    EXPECT_EQ(var,0x96D);
    var = 0;
    ifs.assignBitPattern<std::uint64_t>(var,0,0x1,1);
    EXPECT_EQ(var,1);
    ifs.assignBitPattern<std::uint64_t>(var,1,0x1,1);
    EXPECT_EQ(var,2);
    */
}

TEST(informationset_test, writeSingleOnNonParam_test)
{
    crazyara::InformationSet<bitSize> ifs;
    std::vector<std::bitset<bitSize>> bitsets(32,random_bitset<bitSize>(0.5));
    for(std::bitset<bitSize>& bitset : bitsets)
        ifs.add(bitset);
    EXPECT_EQ(ifs.size(), 32);
}

TEST(informationset_test, writeSingleOnZeroSize_test)
{
    crazyara::InformationSet<bitSize> ifs(0);
    std::vector<std::bitset<bitSize>> bitsets(32,random_bitset<bitSize>(0.5));
    for(std::bitset<bitSize>& bitset : bitsets)
        ifs.add(bitset);
    EXPECT_EQ(ifs.size(), 32);
}

TEST(informationset_test, writeSingleOnNonZeroSize_test)
{   
    crazyara::InformationSet<bitSize> ifs(5);
    std::vector<std::bitset<bitSize>> bitsets(32,random_bitset<bitSize>(0.5));
    for(std::bitset<bitSize>& bitset : bitsets)
        ifs.add(bitset);
    EXPECT_EQ(ifs.size(), 32);
}

/*
TEST(informationset_test, readWriteIteratorComp_test)
{   
    crazyara::InformationSet<bitSize> ifs;
    std::vector<std::bitset<bitSize>> bitsets(2,random_bitset<bitSize>(0.5));
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
    std::cout<<"Bits:"<<bitsets[0]<<std::endl;
    ifs.add(bitsets);
    EXPECT_EQ(ifs.size(),bitsets.size());
    
    for(int i=0; i<bitsets.size(); i++)
    {
        std::unique_ptr<std::bitset<bitSize>> data = ifs.getBitPattern(i);
        EXPECT_EQ(*data,bitsets[i]);
    }
}
*/
};

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}





