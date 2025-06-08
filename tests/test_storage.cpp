#include <gtest/gtest.h> 

#include "Storage.hpp"

TEST(Test, Copy) { 
    Storage storage(10, dsizeof(float32));
    ASSERT_EQ(storage.references(), 1);

    float* stored = static_cast<float*>(storage.address()); 
    for (int index = 0; index < 10; ++index) {
        stored[index] = static_cast<float>(index) * 1.5f;
    }

    {
        Storage reference = storage;
        ASSERT_EQ(storage.references(), 2);
        ASSERT_EQ(reference.references(), 2);

        float* referenced = static_cast<float*>(reference.address());
        for (int index = 0; index < 10; ++index) {
            ASSERT_EQ(referenced[index], stored[index]);
        }
    } // <-- 'reference' goes out of scope

    ASSERT_EQ(storage.references(), 1);
}

TEST(Test, Move) { 
    Storage storage(10, dsizeof(float32));
    ASSERT_EQ(storage.references(), 1);
    float* stored = static_cast<float*>(storage.address()); 
    for (int index = 0; index < 10; ++index) {
        stored[index] = static_cast<float>(index) * 1.5f;
    }
    Storage same = std::move(storage);

    float* moved = static_cast<float*>(same.address());
    for (int index = 0; index < 10; ++index) {
        ASSERT_EQ(moved[index], static_cast<float>(index) * 1.5f);
    }
    ASSERT_EQ(storage.references(), 0); // or check for nullptr if exposed
}

TEST(Test, SelfAssignment) {
    Storage storage(10, dsizeof(float32));
    storage = storage; // should be a no-op
    ASSERT_EQ(storage.references(), 1);
}

TEST(Test, MultipleCopies) {
    Storage storage(10, dsizeof(float32));
    {
        Storage b = storage;
        Storage c = storage;
        ASSERT_EQ(storage.references(), 3);
    } // b and c go out of scope
    ASSERT_EQ(storage.references(), 1);
}