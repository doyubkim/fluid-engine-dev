// Copyright (c) 2016 Doyub Kim

#include <cassert>

namespace jet
{

    template <std::size_t N>
    Size<N>::Size()
    {
        for (auto& elem : elements)
        {
            elem = 0;
        }
    }

    template <std::size_t N>
    template <typename... Params>
    Size<N>::Size(Params... params)
    {
        static_assert(sizeof...(params) == N, "Invalid number of parameters.");

        setAt(0, params...);
    }

    template <std::size_t N>
    Size<N>::Size(const std::initializer_list<std::size_t>& lst)
    {
        assert(lst.size() >= N);

        std::size_t i = 0;
        for (const auto& inputElem : lst)
        {
            elements[i] = inputElem;
            ++i;
        }
    }

    template <std::size_t N>
    Size<N>::Size(const Size& other) :
        elements(other.elements)
    {
    }

    template <std::size_t N>
    const std::size_t& Size<N>::operator[](std::size_t i) const
    {
        return elements[i];
    }

    template <std::size_t N>
    std::size_t& Size<N>::operator[](std::size_t i)
    {
        return elements[i];
    }

    template <std::size_t N>
    template <typename... Params>
    void Size<N>::setAt(std::size_t i, std::size_t v, Params... params)
    {
        elements[i] = v;

        setAt(i+1, params...);
    }

    template <std::size_t N>
    void Size<N>::setAt(std::size_t i, std::size_t v)
    {
        elements[i] = v;
    }

}

