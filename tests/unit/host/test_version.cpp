#include <gtest/gtest.h>
#include <truellm/version.h>

#include <string>

TEST(VersionTest, ComponentsAreNonNegative)
{
    EXPECT_GE(truellm::Version::major, 0);
    EXPECT_GE(truellm::Version::minor, 0);
    EXPECT_GE(truellm::Version::patch, 0);
}

TEST(VersionTest, AbiVersionIsOneForInitialRelease)
{
    EXPECT_EQ(truellm::Version::abi, 1);
}

TEST(VersionTest, StringMatchesComponents)
{
    std::string expected = std::to_string(truellm::Version::major) + "."
                         + std::to_string(truellm::Version::minor) + "."
                         + std::to_string(truellm::Version::patch);
    EXPECT_EQ(std::string(truellm::Version::string), expected);
}
