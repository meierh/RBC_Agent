/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: informationset.h
 * Created on 17.10.2023
 * @author: meierh
 *
 * Abstract class for a information set.
 */

#ifndef INFORMATIONSET_H
#define INFORMATIONSET_H

#include <gtest/gtest.h>
#include <iostream>
#include <cassert>
#include <memory>
#include <cstdint>
#include <bitset>
#include <vector>
#include <set>

namespace crazyara {
/**
 * @brief The InformationSet class defines a generic information set interface to store a game state.
 */
template<std::uint64_t numberBitsPerItem>
class InformationSet
{
    public:
        InformationSet(std::uint64_t initialCapacity=1):
        infoSetSize(0),
        infoSetCapacity(initialCapacity)
        {
            numberBytesPerItem = numberBitsPerItem / (sizeof(std::uint8_t)*8);
            unusedBitsPerItem = sizeof(std::uint8_t)*8 - (numberBitsPerItem % (sizeof(std::uint8_t)*8));
            unusedBitsPerItem %= sizeof(std::uint8_t)*8;
            if(unusedBitsPerItem != 0)
                numberBytesPerItem++;
            
            if(infoSetCapacity<1)
                infoSetCapacity=1;
    
            infoSet = std::make_unique<std::uint8_t[]>(numberBytesPerItem*infoSetCapacity);
            //std::cout<<"Constructed"<<std::endl;
        };
        
        std::uint64_t size() const
        {
            return infoSetSize;            
        };
        
        void add(const std::bitset<numberBitsPerItem>& item)
        {
            //std::cout<<"Add single"<<std::endl;
            const std::vector<std::bitset<numberBitsPerItem>> items = {item};
            add(items);
        };
        
        void add(const std::vector<std::bitset<numberBitsPerItem>>& items)
        {
            //std::cout<<"Add multi"<<std::endl;
            std::uint64_t validItemsNumber = infoSetSize-removedBoards.size();
            bool expandInfoSet=false;
            while(validItemsNumber+items.size() > infoSetCapacity)
            {
                infoSetCapacity*=2;
                expandInfoSet=true;
            }
            //std::cout<<"Expand:"<<expandInfoSet<<std::endl;
            if(expandInfoSet)
            {
                std::unique_ptr<std::uint8_t[]> prevInfoSet = std::move(infoSet);
                infoSet = std::make_unique<std::uint8_t[]>(numberBytesPerItem*infoSetCapacity);
                std::uint64_t newInfoSetIndex=0;
                for(std::uint64_t index=0; index<infoSetSize; index++)
                {
                    if(removedBoards.find(index)==removedBoards.end())
                    {
                        for(std::uint64_t byteInd=0; byteInd<numberBytesPerItem; byteInd++)
                        {
                            infoSet[newInfoSetIndex+byteInd] = prevInfoSet[index+byteInd];
                        }
                        newInfoSetIndex++;
                    }
                }
                removedBoards.clear();
                for(std::uint64_t index=0; index<items.size(); index++,newInfoSetIndex++)
                {
                    setBitPattern(newInfoSetIndex,items[index]);
                }
                infoSetSize = newInfoSetIndex;
            }
            else
            {
                std::uint64_t index=0;
                auto iter = removedBoards.begin();
                while(iter != removedBoards.end())
                {
                    std::uint64_t infoSetIndex = *iter;
                    setBitPattern(infoSetIndex,items[index]);
                    index++;
                    iter = removedBoards.erase(iter);
                }
                std::uint64_t newInfoSetIndex = infoSetSize;
                for(; index<items.size(); index++,newInfoSetIndex++)
                {
                    setBitPattern(newInfoSetIndex,items[index]);
                }
                infoSetSize = newInfoSetIndex;
            }
        };
        
        void remove(const std::uint64_t itemInd)
        {
            removedBoards.insert(itemInd);
        };
        
        void remove(const std::vector<std::uint64_t>& itemsInd)
        {
            for(const std::uint64_t itemInd : itemsInd)
                remove(itemInd);
        };
        
        void clearRemoved(bool resetCapacity=true)
        {
            std::uint64_t validItemsNumber = infoSetSize-removedBoards.size();
            bool reducedInfoSet=false;
            std::uint64_t provInfoSetCapacity = infoSetCapacity;
            std::uint64_t provInfoSetCapacityHalf = provInfoSetCapacity/2;
            if(resetCapacity)
            {
                while(provInfoSetCapacityHalf>validItemsNumber)
                {
                    provInfoSetCapacity = provInfoSetCapacityHalf;
                    provInfoSetCapacityHalf/=2;
                    reducedInfoSet=true;
                }
                assert(provInfoSetCapacity>=validItemsNumber);
                infoSetCapacity = provInfoSetCapacity;
            }

            if(reducedInfoSet)
            {
                std::unique_ptr<std::uint8_t[]> prevInfoSet = infoSet;
                infoSet = std::make_unique<std::uint8_t[]>(numberBytesPerItem*infoSetCapacity);
                std::uint64_t newInfoSetIndex=0;
                for(std::uint64_t index=0; index<infoSetSize; index++)
                {
                    if(removedBoards.find(index)==removedBoards.end())
                    {
                        for(std::uint64_t byteInd=0; byteInd<numberBytesPerItem; byteInd++)
                        {
                            infoSet[newInfoSetIndex+byteInd] = prevInfoSet[index+byteInd];
                        }
                        newInfoSetIndex++;
                    }
                }
                removedBoards.clear();
                assert(validItemsNumber==validItemsNumber);
                infoSetSize = newInfoSetIndex;
            }
            else
            {
                std::uint64_t newInfoSetIndex=0;
                for(std::uint64_t index=0; index<infoSetSize; index++)
                {
                    if(removedBoards.find(index)==removedBoards.end())
                    {
                        for(std::uint64_t byteInd=0; byteInd<numberBytesPerItem; byteInd++)
                        {
                            infoSet[newInfoSetIndex+byteInd] = infoSet[index+byteInd];
                        }
                        newInfoSetIndex++;
                    }
                }
                removedBoards.clear();
            }
            infoSetSize = validItemsNumber;
        }
        
    protected:
        std::unique_ptr<std::vector<std::uint8_t>> getRawBitPattern(std::uint64_t index) const
        {
            assert(index<infoSetSize);
            auto bytes = std::make_unique<std::vector<std::uint8_t>>(numberBitsPerItem);
            std::uint64_t startByte = index*numberBytesPerItem;
            std::uint64_t ind=0;
            for(std::uint64_t index = startByte; index<startByte+numberBytesPerItem; index++)
            {
                (*bytes)[ind] = infoSet[index];
                ind++;
            }
            return bytes;    
        };
        
        std::unique_ptr<std::bitset<numberBitsPerItem>> getBitPattern(std::uint64_t index) const
        {
            auto bitPattern = std::make_unique<std::bitset<numberBitsPerItem>>();
            std::unique_ptr<std::vector<std::uint8_t>> bytes = getRawBitPattern(index);
            std::vector<std::uint8_t>& item = *bytes;
            
            std::uint64_t byteInd;
            std::uint64_t bitInd;
            for(std::uint64_t bitPatInd=0; bitPatInd<bitPattern->size(); bitPatInd++)
            {
                std::uint64_t totalInd = bitPatInd+unusedBitsPerItem+1;
                byteInd = totalInd / sizeof(std::uint8_t)*8;
                bitInd = totalInd % sizeof(std::uint8_t)*8;
                (*bitPattern)[bitPatInd] = getBit<std::uint8_t>(item[byteInd],7-bitInd);
            }
            return bitPattern;
        };
        
        void setBitPattern(std::uint64_t index,const std::bitset<numberBitsPerItem>& bitPattern)
        {
            std::vector<bool> bits(bitPattern.size()+unusedBitsPerItem,false);
            for(unsigned int bitInd=0; bitInd<bitPattern.size(); bitInd++)
                bits[bitInd+unusedBitsPerItem] = bitPattern[bitInd];
            
            std::uint64_t startByte = index*numberBytesPerItem;
            std::uint64_t ind=0;
            for(std::uint64_t index = startByte; index<startByte+numberBytesPerItem; index++,ind+=8)
            {
                std::bitset<8> byteSet;
                for(int i=0;i<8;i++)
                    byteSet[i] = bits[i+ind];
                infoSet[index] = transferBytePattern(byteSet,0,8);
            }
        };
        
        /* Set the bit on variable with leftShift counted as 
         * [...,4,3,2,1,0] 
         */
        template<typename T>
        void setBit(T& variable,unsigned int leftShift) const
        {
            if(leftShift>30)
                throw std::invalid_argument("set Bit must be limited to <= 30");
            variable |= (1<<leftShift);
        };

        /* Unset the bit on variable with leftShift counted as 
         * [...,4,3,2,1,0] 
         */
        template<typename T>
        void unsetBit(T& variable,unsigned int leftShift) const
        {
            if(leftShift>30)
                throw std::invalid_argument("unset Bit must be limited to <= 30");
            variable &= ~(1<<leftShift);
        };

        /* Get the bit on variable with leftShift counted as 
         * [...,4,3,2,1,0] 
         */
        template<typename T>
        bool getBit(const T variable,unsigned int leftShift) const
        {
            if(leftShift>30)
                throw std::invalid_argument("get Bit must be limited to <= 30");
            return variable & (1<<leftShift);
        };

        /*
        template<typename T>
        void assignBitPattern(T& variable,unsigned int leftShift,T value,unsigned int size) const
        {
            for(unsigned int localShift=0; localShift<size; localShift++)
            {
                T one = 1;
                one <<= localShift;
                if(value & one)
                {
                    setBit(variable,leftShift+localShift);
                }
                else
                {
                    unsetBit(variable,leftShift+localShift);
                }
            }
        };
        */
        
        /*
        template<typename T>
        T extractBitPattern(T variable,unsigned int leftShift,unsigned int size) const
        {
            T ones = 1;
            for(unsigned int i=1;i<size;i++)
            {
                ones += (ones+1); 
            }
            variable >>= leftShift;
            return variable & ones;
        };
        */
        
        template<typename T>
        void assignBitPattern
        (
            std::bitset<numberBitsPerItem>& variable,
            unsigned int startIndex,
            T value,
            unsigned int size
        ) const
        {
            for(unsigned int i=0; i<size; i++)
            {
                variable[i+startIndex] = getBit(value,size-1-i);
            }
        };
        
        template<typename T>
        T transferBitPattern
        (
            const std::bitset<numberBitsPerItem>& variable,
            unsigned int startIndex,
            unsigned int size
        ) const
        {
            T result = 0;
            for(unsigned int i=0; i<size; i++)
            {
                if(variable[i+startIndex])
                {
                    setBit(result,size-1-i);
                }
                else
                {
                    unsetBit(result,size-1-i);
                }
            }
            return result;
        };
        
        std::uint8_t transferBytePattern
        (
            const std::bitset<8>& variable,
            unsigned int startIndex,
            unsigned int size
        ) const
        {
            std::uint8_t result = 0;
            for(unsigned int i=0; i<size; i++)
            {
                if(variable[i+startIndex])
                {
                    setBit(result,size-1-i);
                }
                else
                {
                    unsetBit(result,size-1-i);
                }
            }
            return result;
        };
        
        friend class IS_Iterator;

        class IS_Iterator
        {
            public:
                IS_Iterator
                (
                    InformationSet<numberBitsPerItem>* is,
                    std::uint64_t itemInd
                )
                {
                    this->is = is;
                    current_Item = itemInd;
                    if(current_Item>=is->infoSetSize)
                        end=true;
                    else
                        end=false;
                };
                
                IS_Iterator
                (
                    InformationSet<numberBitsPerItem>* is
                )
                {
                    std::uint64_t startItem = 0;
                    while(is->removedBoards.find(startItem)!=is->removedBoards.end())
                        startItem++;
                    *this = IS_Iterator(is,startItem);
                };

                IS_Iterator &operator++() noexcept
                {
                    if((current_Item+1)>=is->infoSetSize)
                    {
                        end=true;
                        return *this;
                    }
                    else
                    {
                        current_Item++;
                        while(is->removedBoards.find(current_Item)!=is->removedBoards.end())
                            current_Item++;
                    
                        if(current_Item>=is->infoSetSize)
                            end=true;

                        return *this;
                    }
                };

                IS_Iterator operator++(int) noexcept
                {
                    IS_Iterator tempIter = *this; // we make a copy of the iterator
                    ++*this;                   // we increment
                    return tempIter;           // we return the copy before increment
                };

                bool operator==(const IS_Iterator &other) const noexcept
                {
                    if(this->end==true || other.end==true)
                    {
                        if(other.end==this->end)
                            return true;
                        else
                            return false;
                    }
                    else
                    {
                        if(other.end!=this->end)
                        {
                            return false;
                        }
                        else
                        {
                            return this->current_Item==other.current_Item;
                        }
                    }
                };
                
                bool operator!=(const IS_Iterator &other) const noexcept
                {
                    return !operator==(other);
                };

                std::unique_ptr<std::bitset<numberBitsPerItem>> operator*() const noexcept
                {
                    return is->getBitPattern(current_Item);
                };
                
                std::uint64_t getCurrentIndex() const {return current_Item;};

            protected:
                InformationSet<numberBitsPerItem>* is;
                
            private:
                std::uint64_t current_Item;
                bool end;
        };
            
        IS_Iterator cbegin() noexcept
        {
            return IS_Iterator(this);
        };
        
        IS_Iterator cend() noexcept
        {
            return IS_Iterator(this,infoSetSize);
        };
        
        IS_Iterator remove(IS_Iterator iter)
        {
            removedBoards.insert(iter.getCurrentIndex());
            return ++iter;
        };
        
    private:
        std::unique_ptr<std::uint8_t[]> infoSet;
        std::uint64_t infoSetSize; //Last index in infoSet with a valid item
        std::uint64_t infoSetCapacity; //Total length of infoSet
        std::uint64_t numberBytesPerItem;
        std::uint64_t unusedBitsPerItem;
        std::set<std::uint64_t> removedBoards;
        
        FRIEND_TEST(informationset_test, setBit_test);
        FRIEND_TEST(informationset_test, unsetBit_test);
        FRIEND_TEST(informationset_test, getBit_test);
        FRIEND_TEST(informationset_test, assignBitPattern_test);

        FRIEND_TEST(informationset_test, readWriteIteratorComp_test);
        FRIEND_TEST(informationset_test, readWriteIndexComp_test);
};
}

#endif // INFORMATIONSET_H
