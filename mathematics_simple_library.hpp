#pragma once

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <numeric>
#include <numbers>
#include <complex>
#include <vector>
#include <array>
#include <set>
#include <iterator>
#include <cstddef>
#include <span>
#include <initializer_list>
#include <concepts>
#include <utility>

#ifndef MATHEMATICS_SIMPLE_LIBRARY_NAMESPACE
#define MATHEMATICS_SIMPLE_LIBRARY_NAMESPACE Maths
#endif

#ifndef MATHEMATICS_SIMPLE_LIBRARY_INDEX_TYPE
#define MATHEMATICS_SIMPLE_LIBRARY_INDEX_TYPE std::size_t
#endif

#ifndef MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_COLUMN_MAJOR
#define MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_COLUMN_MAJOR false
#endif

#ifndef MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_NUMERICAL_TYPE
#define MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_NUMERICAL_TYPE float
#endif

#ifndef MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_CONVENTION_RAY_DIRECTION
#define MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_CONVENTION_RAY_DIRECTION Conventions::RayDirection::Outgoing
#endif

namespace MATHEMATICS_SIMPLE_LIBRARY_NAMESPACE {

    namespace Conventions {
        enum class RayDirection {
            Incoming,
            Incident = Incoming,
            Outgoing,
            Reflected = Outgoing
        };
    }

    namespace Units {
        enum class Angular {
            Radians,
            Degrees,
            Turns,
            Gradians,
            Gons = Gradians
            //HourAngle,
            //BinaryDegree
        };
    }

    using IndexType = MATHEMATICS_SIMPLE_LIBRARY_INDEX_TYPE;
    using NumericalTypeDefault = MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_NUMERICAL_TYPE;
    constexpr Conventions::RayDirection RayDirectionDefault = MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_CONVENTION_RAY_DIRECTION;
    constexpr bool ColumnMajorDefault = MATHEMATICS_SIMPLE_LIBRARY_DEFAULT_COLUMN_MAJOR;

    template <bool> struct is_boolean_t {};

    template <typename T>
    struct is_complex_t : public std::false_type {};
    template <typename T>
    struct is_complex_t<std::complex<T>> : public std::true_type {};
    template <typename T>
    inline constexpr bool is_complex_v = is_complex_t<T>::value;

    template <class T>
    struct is_array_t : public std::false_type {};
    template <class T, std::size_t N>
    struct is_array_t<std::array<T, N>> : public std::true_type {};
    template <class T>
    inline constexpr bool is_array_v = is_array_t<T>::value;

    template <class T, class = void> struct value_type { using type = T; };
    template <class T> struct value_type<T, std::void_t<typename T::value_type>> { using type = typename T::value_type; };
    template <class T> using value_type_t = typename value_type<T>::type;

    template<class T, T... Indices, class F>
    constexpr void static_loop(std::integer_sequence<T, Indices...>, F&& f) {
        (f(std::integral_constant<T, Indices>{}), ...);
    }

    template<class T, T Count, class F>
    constexpr void static_loop(F&& f) {
        static_loop(std::make_integer_sequence<T, Count>{}, std::forward<F>(f));
    }

    template <typename T>
    constexpr bool is_power_of_two(T x) { return x && ((x & (x-1)) == 0); }

    template <Units::Angular SourceUnit, Units::Angular TargetUnit, typename T>
    constexpr auto convert_units_angular(T x) {
        if constexpr(TargetUnit == Units::Angular::Radians) {
            if constexpr(SourceUnit == Units::Angular::Radians) return x;
            else if constexpr(SourceUnit == Units::Angular::Degrees) return x*std::numbers::pi_v<T>/T{180};
            else if constexpr(SourceUnit == Units::Angular::Turns) return x*T{2}*std::numbers::pi_v<T>;
            else if constexpr(SourceUnit == Units::Angular::Gradians) return x*std::numbers::pi_v<T>/T{200};
        }
        else if constexpr(TargetUnit == Units::Angular::Degrees) {
            if constexpr(SourceUnit == Units::Angular::Radians) return x*T{180}/std::numbers::pi_v<T>;
            else if constexpr(SourceUnit == Units::Angular::Degrees) return x;
            else if constexpr(SourceUnit == Units::Angular::Turns) return x*T{360};
            else if constexpr(SourceUnit == Units::Angular::Gradians) return x*T{9}/T{10};
        }
        else if constexpr(TargetUnit == Units::Angular::Turns) {
            if constexpr(SourceUnit == Units::Angular::Radians) return x/(T{2}*std::numbers::pi_v<T>);
            else if constexpr(SourceUnit == Units::Angular::Degrees) return x/T{360};
            else if constexpr(SourceUnit == Units::Angular::Turns) return x;
            else if constexpr(SourceUnit == Units::Angular::Gradians) return x/T{400};
        }
        else if constexpr(TargetUnit == Units::Angular::Gradians) {
            if constexpr(SourceUnit == Units::Angular::Radians) return x*T{200}/(std::numbers::pi_v<T>);
            else if constexpr(SourceUnit == Units::Angular::Degrees) return x*T{10}/T{9};
            else if constexpr(SourceUnit == Units::Angular::Turns) return x*T{400};
            else if constexpr(SourceUnit == Units::Angular::Gradians) return x;
        }
    }

    template <typename T, Units::Angular SourceUnit = Units::Angular::Radians>
    constexpr auto degrees(T x) {
        return convert_units_angular<SourceUnit, Units::Angular::Degrees>(x);
    }
    template <Units::Angular SourceUnit, typename T>
    constexpr auto degrees(T x) { degrees<T, SourceUnit>(x); }
    
    template <typename T, Units::Angular SourceUnit = Units::Angular::Degrees>
    constexpr auto radians(T x) {
        return convert_units_angular<SourceUnit, Units::Angular::Radians>(x);
    }
    template <Units::Angular SourceUnit, typename T>
    constexpr auto radians(T x) { radians<T, SourceUnit>(x); }

    template <class T>
    concept ConceptContainer = requires(T& container) {
        typename T::value_type;
        {std::data(container)};
        {std::size(container)};
    };

    template <class T>
    concept ConceptIterable = requires(T& object) {
        {std::begin(object)};
        {std::end(object)};
    };

    template <typename E>
    concept ConceptExtent = requires (const E& extent) {
        { extent.get() } -> std::convertible_to<IndexType>;
        typename is_boolean_t<E::is_static()>;
    };

    template <IndexType KnownSize>
    struct StaticExtent {
        constexpr StaticExtent([[maybe_unused]] IndexType value = KnownSize) {
            assert(value == KnownSize);
        }

        template <IndexType OtherSize>
        constexpr auto operator+ (const StaticExtent<OtherSize>&) const {
            return StaticExtent<KnownSize + OtherSize> {};
        }

        template <IndexType OtherSize>
        constexpr auto operator- (const StaticExtent<OtherSize>&) const {
            return StaticExtent<KnownSize - OtherSize> {};
        }

        template <IndexType OtherSize>
        constexpr auto operator* (const StaticExtent<OtherSize>&) const {
            return StaticExtent<KnownSize * OtherSize> {};
        }

        template <IndexType OtherSize>
        constexpr auto operator/ (const StaticExtent<OtherSize>&) const {
            return StaticExtent<KnownSize / OtherSize> {};
        }

        static constexpr IndexType get() { return KnownSize; } 
        static constexpr bool is_static() { return true; }
    };

    struct DynamicExtent {
        IndexType value;

        DynamicExtent(IndexType value) : value(value) {}

        DynamicExtent operator+ (const DynamicExtent& other) const {
            return DynamicExtent { get() + other.get() };
        }

        DynamicExtent operator- (const DynamicExtent& other) const {
            return DynamicExtent { get() - other.get() };
        }

        DynamicExtent operator* (const DynamicExtent& other) const {
            return DynamicExtent { get() * other.get() };
        }

        DynamicExtent operator/ (const DynamicExtent& other) const {
            return DynamicExtent { get() / other.get() };
        }

        constexpr IndexType get() const { return value; }
        static constexpr bool is_static() { return false; }
    };

    template <ConceptExtent E, ConceptExtent F, typename ComparisonOperator>
    requires (E::is_static() && F::is_static())
    constexpr void assert_extent(const E&, const F&, ComparisonOperator cmp) {
        static_assert(cmp(E::get(), F::get()));
    }

    template <ConceptExtent E, ConceptExtent F, typename ComparisonOperator>
    void assert_extent(
        [[maybe_unused]] const E& extent,
        [[maybe_unused]] const F& other,
        [[maybe_unused]] ComparisonOperator cmp
    ) { assert(cmp(extent.get(), other.get())); }

    template <ConceptExtent E, ConceptExtent F, typename ComparisonOperator>
    constexpr auto compare_extent(
        [[maybe_unused]] const E& extent,
        [[maybe_unused]] const F& other,
        [[maybe_unused]] ComparisonOperator cmp
    ) { return cmp(extent.get(), other.get()); }

    template <IndexType Value, ConceptExtent E, typename BinaryOperator>
    constexpr auto evaluate_extent([[maybe_unused]] const E& extent, [[maybe_unused]] BinaryOperator op) {
        if constexpr(E::is_static())
            return op(extent, StaticExtent<Value>{});
        else
            return op(extent, Value);
    }

    template <ConceptExtent E, ConceptExtent F, typename BinaryOperator>
    constexpr auto evaluate_extent([[maybe_unused]] const E& extent, [[maybe_unused]] const F& other, [[maybe_unused]] BinaryOperator op) {
        if constexpr((E::is_static() && F::is_static()) || !(E::is_static() || F::is_static()))
            return op(extent, other);
        else if constexpr(E::is_static() && !F::is_static())
            return op(DynamicExtent{E::get()}, other);
        else if constexpr(!E::is_static() && F::is_static())
            return op(extent, DynamicExtent{F::get()});
    }

    template <typename T>
    concept ConceptDataWrapper = requires(const T& object, IndexType element) {
        { object.data[element] };
    };
    
    template <typename V>
    concept ConceptVector = requires(const V& vector, IndexType element) {
        typename V::value_type;
        { vector[element] };
        { vector.size() } -> ConceptExtent;
    };
    
    template <typename V>
    concept ConceptVectorStatic = ConceptVector<V> && requires(const V& vector) {
        {decltype(vector.size())::get()};
    };

    template <typename V>
    concept ConceptVectorObject = ConceptVector<V> && ConceptDataWrapper<V>;

    template <typename V>
    concept ConceptVectorObjectStatic = ConceptVectorStatic<V> && ConceptDataWrapper<V>;

    template <ConceptVector V>
    constexpr auto size_static() { return decltype(std::declval<V>().size())::get(); }

    struct QuaternionTag {};

    template <typename V>
    concept ConceptQuaternion = ConceptVectorStatic<V> && size_static<V>() == 4 && requires(const V& vector) {
        { std::same_as<typename V::tag_type, const QuaternionTag&> };
    };

    template <typename M>
    concept ConceptMatrix = requires(const M& matrix, IndexType row, IndexType column) {
        typename M::value_type;
        { M::column_major } -> std::same_as<const bool&>;
        { matrix.operator[](row, column) };
        { matrix.row_count() } -> ConceptExtent;
        { matrix.column_count() } -> ConceptExtent;
    };

    template <typename M>
    concept ConceptMatrixStatic = ConceptMatrix<M> && requires(const M& matrix) {
        {decltype(matrix.row_count())::get()};
        {decltype(matrix.column_count())::get()};
    };

    template <typename M>
    concept ConceptMatrixObject = ConceptMatrix<M> && ConceptDataWrapper<M>;

    template <typename M>
    concept ConceptMatrixObjectStatic = ConceptMatrixStatic<M> && ConceptDataWrapper<M>;

    template <ConceptMatrix M>
    constexpr auto row_count_static() { return decltype(std::declval<M>().row_count())::get(); }
    template <ConceptMatrix M>
    constexpr auto column_count_static() { return decltype(std::declval<M>().column_count())::get(); }

    //NOTE: may return negative zero
    template <typename T>
    requires (!ConceptVector<T> && !ConceptMatrix<T>)
    constexpr T euclidean_remainder(T a, T b) {
        assert(b != T(0));
        T result;
        if constexpr(std::is_integral_v<T>) {
            result = a % b;
        } else {
            result = std::fmod(a,b);
        }
        if constexpr(!std::is_unsigned_v<T>) b = std::abs(b);
        return result >= T(0) ? result : result + b;
    }
    template <typename T, typename U>
    constexpr auto euclidean_modulo(T a, U b) { return euclidean_remainder(a, b); }
    template <typename T, typename U>
    constexpr auto eucmod(T a, U b) { return euclidean_remainder(a, b); }

    template <typename T>
    constexpr auto circular_shift(T x, T shift, T width) { return euclidean_modulo(x + shift, width); }
    template <typename T>
    constexpr auto circshift(T x, T shift, T width) { return circular_shift(x, shift, width); }

    template <typename T>
    constexpr auto fftshift(T x, T width) {
        if constexpr(std::is_unsigned_v<T>)
            return eucmod(width - eucmod(x, width) + width/static_cast<T>(2), width);
        else
            return circular_shift(-x, width/static_cast<T>(2), width);
    }

    template <typename A, typename B, typename T>
    constexpr auto linear_interpolation(const A& a, const B& b, const T& t) {
        return a*(T{1} - t) + b*t;
    }
    template <typename A, typename B, typename T>
    constexpr auto lerp(const A& a, const B& b, const T& t) { return linear_interpolation(a, b, t); }

    template <typename T>
    constexpr T kronecker_delta(IndexType i, IndexType j) { return static_cast<T>(i==j); }

    template <typename T>
    constexpr auto conjugate(const T& x) {
        if constexpr(is_complex_v<T>)
            return std::conj(x);
        else if constexpr(std::is_integral_v<T> || std::is_floating_point_v<T>)
            return x;
        else
            return conj(x);
    }
    template <typename T> constexpr auto conj(const T& x) { return conjugate(x); }


    template <ConceptExtent E, typename Field>
    struct VectorConstant {
        Field value;
        E extent = {};

        using value_type = Field;

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            return value;
        }

        constexpr auto size() const { return extent; }
    };

    template <IndexType Size, typename T>
    constexpr auto vec_constant(const T& value) {
        return VectorConstant<StaticExtent<Size>, T> { value };
    }
    template <IndexType Size, typename T>
    constexpr auto as_vector(const T& value) {
        return vec_constant<Size, T>(value);
    }

    template <typename T>
    constexpr auto vec_constant(const T& value, IndexType size) {
        return VectorConstant<DynamicExtent, T> { value, size };
    }
    template <typename T>
    constexpr auto as_vector(const T& value, IndexType size) {
        return vec_constant<T>(value, size);
    }

    template <typename Field, ConceptExtent E>
    struct VectorReference {
        Field* base_ptr;
        E extent;

        using value_type = Field;

        constexpr VectorReference(Field* base_ptr, const E& size = {})
            : base_ptr(base_ptr), extent(size)
        {}

        constexpr auto ref() const { return *this; }

        template <ConceptVector V>
        constexpr void set_from(const V& v) const {
            auto copy_extent = std::min(size().get(), v.size().get());
            for (IndexType i = 0; i < copy_extent; ++i)
                if constexpr (ConceptIterable<typename V::value_type>)
                    std::copy(std::begin(v[i]), std::end(v[i]), std::begin((*this)[i]));
                else
                    (*this)[i] = static_cast<typename V::value_type>(v[i]);
        }

        template <ConceptVector V>
        constexpr auto& operator= (const V& v) const {
            assert_extent(v.size(), this->size(), std::greater_equal<>{});
            set_from(v);
            return *this;
        }

        constexpr Field& operator[] (IndexType element) const {
            assert(element < this->size().get());
            return base_ptr[element];
        }

        constexpr auto size() const { return extent; }
    };

    template <typename T>
    constexpr auto vec_ref(T* base_ptr, IndexType size) {
        return VectorReference<T, DynamicExtent> { base_ptr, size };
    }

    template <IndexType Size, typename T>
    constexpr auto vec_ref(T* base_ptr) {
        return VectorReference<T, StaticExtent<Size>> { base_ptr };
    }
    
    template <typename T, std::size_t N>
    constexpr auto vec_ref(const T(&elements)[N]) {
        return VectorReference<const T, StaticExtent<N>> { std::data(elements) };
    }

    template <typename T, IndexType N>
    constexpr auto vec_ref(const std::span<T, N>& span) {
        if constexpr (N == std::dynamic_extent) {
            return vec_ref<T>(span.data(), span.size());
        } else {
            return vec_ref<N, T>(span.data());
        }
    }

    template <typename T, IndexType N>
    constexpr auto vec_ref(const std::array<T, N>& array) {
        return vec_ref(std::span { array });
    }

    template <typename T>
    constexpr auto vec_ref(const std::vector<T>& vector) {
        return vec_ref(std::span { vector });
    }

    template <ConceptContainer T>
    constexpr auto vec_ref(T& container) {
        return vec_ref<typename T::value_type>(std::data(container), std::size(container));
    }

    template <typename T, IndexType Size>
    struct VectorObjectStatic {
        std::array<T, Size> data;

        using value_type = T;

        constexpr VectorObjectStatic() = default;
        constexpr VectorObjectStatic(const VectorObjectStatic&) = default;
        constexpr VectorObjectStatic(const T& value) {
            std::fill(data.begin(), data.end(), value);
        }

        template <ConceptVectorStatic V>
        requires (Size <= size_static<V>())
        constexpr VectorObjectStatic(const V& v) { *this = v; }

        template <ConceptVector V>
        requires (!ConceptVectorStatic<V>)
        constexpr VectorObjectStatic(const V& v) {
            assert(this->size().get() <= v.size().get());
            *this = v;
        }

        template <typename... Values>
        requires (sizeof...(Values) == Size)
        constexpr VectorObjectStatic(Values... values) {
            data = {static_cast<T>(values)...};
        }

        template <typename U>
        using RefType = VectorReference<U, StaticExtent<Size>>;

        constexpr auto ref() const { return RefType<const T> { std::data(data) }; }
        auto ref() { return RefType<T> { std::data(data) }; }

        template <ConceptVector V>
        constexpr VectorObjectStatic& operator= (const V& v) { ref() = v; return *this; }

        auto& operator[] (IndexType element) { return ref()[element]; }
        constexpr auto& operator[] (IndexType element) const { return ref()[element]; }
        constexpr auto size() const { return ref().size(); }
    };

    template <typename T>
    struct VectorObjectDynamic {
        std::vector<T> data;

        using value_type = T;

        VectorObjectDynamic()
            : VectorObjectDynamic(0, 0)
        {}

        VectorObjectDynamic(const VectorObjectDynamic&) = default;
        VectorObjectDynamic(VectorObjectDynamic&& other) { *this = std::move(other); }

        template <ConceptVector V>
        VectorObjectDynamic(const V& v) { *this = v; }
        VectorObjectDynamic(IndexType size) : data(size) {}

        VectorObjectDynamic(std::initializer_list<T> elements)
            : VectorObjectDynamic(elements.size())
        {
            std::copy(elements.begin(), elements.end(), data.begin());
        }

        void resize(IndexType size) { data.resize(size); }

        VectorObjectDynamic& operator= (VectorObjectDynamic&& other) {
            data = std::move(other.data);
            return *this;
        }

        VectorObjectDynamic& operator= (const VectorObjectDynamic& other) {
            return (*this).template operator= <VectorObjectDynamic>(other);
        }

        template <ConceptVector V>
        VectorObjectDynamic& operator= (const V& v) {
            resize(v.size().get());
            ref() = v;
            return *this;
        }

        template <typename U>
        using RefType = VectorReference<U, DynamicExtent>;

        auto ref() const { return RefType<const T> {std::data(data), std::size(data)}; }
        auto ref() { return RefType<T> { std::data(data), std::size(data) }; }

        auto& operator[] (IndexType element) { return ref()[element]; }
        const auto& operator[] (IndexType element) const { return ref()[element]; }
        auto size() const { return ref().size(); }
    };

    template <typename... Values>
    constexpr auto vec(Values... values) {
        return VectorObjectStatic<std::common_type_t<Values...>, sizeof...(Values)>(values...);
    }

    template <IndexType Size, ConceptContainer T>
    inline auto vec(const T& container) {
        return VectorObjectStatic<T, Size>({container});
    }

    template <IndexType Size, ConceptVector V>
    inline auto vec(const V& v) {
        return VectorObjectStatic<std::remove_const_t<typename V::value_type>, Size>(v);
    }

    template <typename T>
    inline auto vec(std::initializer_list<T> elements) {
        return VectorObjectDynamic<T>(elements);
    }

    template <ConceptContainer T>
    inline auto vec(const T& container) {
        return VectorObjectDynamic<T>(container);
    }

    template <ConceptVectorStatic V>
    inline auto vec(const V& v) {
        return VectorObjectStatic<std::remove_const_t<typename V::value_type>, size_static<V>()>(v);
    }

    template <ConceptVector V>
    inline auto vec(const V& v) {
        return VectorObjectDynamic<std::remove_const_t<typename V::value_type>>(v);
    }

    template <typename T, ConceptVectorStatic V>
    inline auto vec(const V& v) {
        return VectorObjectStatic<T, size_static<V>()>(v);
    }

    template <typename T, ConceptVector V>
    inline auto vec(const V& v) {
        return VectorObjectDynamic<T>(v);
    }

    template <IndexType Size, typename T>
    using vec_static_t = VectorObjectStatic<T, Size>;

    template <typename T>
    using vec_dynamic_t = VectorObjectDynamic<T>;

    template <ConceptVector L, ConceptVector R>
    struct ConcatenatedVector {
        L left;
        R right;

        using value_type = typename L::value_type;

        constexpr ConcatenatedVector(const L& l, const R& r)
            : left(l), right(r)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            if(element >= left.size().get())
                return static_cast<value_type>(right[element - left.size().get()]);
            return left[element];
        }

        constexpr auto size() const { return evaluate_extent(left.size(), right.size(), std::plus<>{}); }
    };

    template <ConceptVector L, ConceptVector R>
    constexpr auto join(const L& l, const R& r) {
        return ConcatenatedVector<L, R>(l.ref(), r.ref());
    }
    template <ConceptVector L, typename R>
    constexpr auto join(const L& l, const R& r) { return join(l, vec_constant<1>(r)); }
    template <typename L, ConceptVector R>
    constexpr auto join(const L& l, const R& r) { return join(vec_constant<1>(l), r); }
    template <typename L, typename R>
    constexpr auto join(const L& l, const R& r) { return join(vec_constant<1>(l), vec_constant<1>(r)); }

    template <typename L, typename R>
    constexpr auto vec(const L& l, const R& r) {
        return vec(join(l, r));
    }
    
    enum class MatrixIdentityType {
        Additive,
        Multiplicative,
        Hadamard
    };

    template <typename Field, ConceptExtent ExtR, ConceptExtent ExtC, MatrixIdentityType Type, bool ColumnMajor>
    struct MatrixIdentity {
        ExtR rows;
        ExtC columns;

        using value_type = Field;
        constexpr static bool column_major = ColumnMajor;

        constexpr MatrixIdentity(const ExtR& rows = {}, const ExtC& columns = {})
            : rows(rows), columns(columns)
        {}

        constexpr auto ref() const { return *this; }

        constexpr Field operator[] ([[maybe_unused]] IndexType row, [[maybe_unused]] IndexType column) const {
            if constexpr(Type == MatrixIdentityType::Additive) {
                return static_cast<Field>(0);
            } else if constexpr(Type == MatrixIdentityType::Multiplicative) {
                return kronecker_delta<Field>(row, column);
            } else if constexpr(Type == MatrixIdentityType::Hadamard) {
                return static_cast<Field>(1);
            }
        }

        constexpr auto row_count() const { return rows; }
        constexpr auto column_count() const { return columns; }
    };

    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_additive_identity() {
        return MatrixIdentity<T, StaticExtent<Rows>, StaticExtent<Columns>, MatrixIdentityType::Additive, ColumnMajor> { {}, {} };
    }
    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_additive_identity(IndexType rows, IndexType columns) {
        return MatrixIdentity<T, DynamicExtent, DynamicExtent, MatrixIdentityType::Additive, ColumnMajor> { rows, columns };
    }
    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_zero() {
        return mat_additive_identity<Rows, Columns, T, ColumnMajor>();
    }
    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_zero(IndexType rows, IndexType columns) {
        return mat_additive_identity<T, ColumnMajor>(rows, columns);
    }

    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_multiplicative_identity() {
        return MatrixIdentity<T, StaticExtent<Rows>, StaticExtent<Columns>, MatrixIdentityType::Multiplicative, ColumnMajor> { {}, {} };
    }
    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_multiplicative_identity(IndexType rows, IndexType columns) {
        return MatrixIdentity<T, DynamicExtent, DynamicExtent, MatrixIdentityType::Multiplicative, ColumnMajor> { rows, columns };
    }
    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_identity() {
        return mat_multiplicative_identity<Rows, Columns, T, ColumnMajor>();
    }
    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_identity(IndexType rows, IndexType columns) {
        return mat_multiplicative_identity<T, ColumnMajor>(rows, columns);
    }

    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_hadamard_identity() {
        return MatrixIdentity<T, StaticExtent<Rows>, StaticExtent<Columns>, MatrixIdentityType::Hadamard, ColumnMajor> { {}, {} };
    }
    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_hadamard_identity(IndexType rows, IndexType columns) {
        return MatrixIdentity<T, DynamicExtent, DynamicExtent, MatrixIdentityType::Hadamard, ColumnMajor> { rows, columns };
    }
    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_one() {
        return mat_hadamard_identity<Rows, Columns, T, ColumnMajor>();
    }
    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_one(IndexType rows, IndexType columns) {
        return mat_hadamard_identity<T, ColumnMajor>(rows, columns);
    }

    template <typename Field, ConceptExtent ExtR, ConceptExtent ExtC, bool ColumnMajor>
    struct MatrixConstant {
        Field value;
        ExtR rows;
        ExtC columns;

        using value_type = Field;
        constexpr static bool column_major = ColumnMajor;

        constexpr MatrixConstant(const Field& value, const ExtR& rows = {}, const ExtC& columns = {})
            : value(value), rows(rows), columns(columns)
        {}

        constexpr auto ref() const { return *this; }

        constexpr Field operator[] ([[maybe_unused]] IndexType row, [[maybe_unused]] IndexType column) const {
            return value;
        }

        constexpr auto row_count() const { return rows; }
        constexpr auto column_count() const { return columns; }
    };

    template <bool ColumnMajor, IndexType Rows, IndexType Columns, typename T>
    constexpr auto mat_constant(const T& value) {
        return MatrixConstant<T, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor> { value };
    }
    template <IndexType Rows, IndexType Columns, typename T, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_constant(const T& value) {
        return mat_constant<ColumnMajor, Rows, Columns, T>(value);
    }
    template <IndexType Rows, IndexType Columns, typename T, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_matrix(const T& value) {
        return mat_constant<ColumnMajor, Rows, Columns, T>(value);
    }

    template <bool ColumnMajor, typename T>
    constexpr auto mat_constant(const T& value, IndexType rows, IndexType columns) {
        return MatrixConstant<T, DynamicExtent, DynamicExtent, ColumnMajor> { value, rows, columns };
    }
    template <typename T, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_constant(const T& value, IndexType rows, IndexType columns) {
        return mat_constant<ColumnMajor, T>(value, rows, columns);
    }
    template <typename T, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_matrix(const T& value, IndexType rows, IndexType columns) {
        return mat_constant<ColumnMajor, T>(value, rows, columns);
    }

    template <typename Field, ConceptExtent ExtR, ConceptExtent ExtC, bool ColumnMajor>
    struct MatrixReference {
        Field* base_ptr;
        ExtR rows;
        ExtC columns;

        using value_type = Field;
        constexpr static bool column_major = ColumnMajor;

        constexpr MatrixReference(Field* base_ptr, const ExtR& rows = {}, const ExtC& columns = {})
            : base_ptr(base_ptr), rows(rows), columns(columns)
        {}

        constexpr auto ref() const { return *this; }

        template <ConceptMatrix M>
        constexpr void set_from(const M& m) const {
            auto
                rows_copy_extent = std::min(row_count().get(), m.row_count().get()),
                column_copy_extent = std::min(column_count().get(), m.column_count().get());
            for (IndexType row = 0; row < rows_copy_extent; ++row)
                for (IndexType column = 0; column < column_copy_extent; ++column)
                    if constexpr (ConceptIterable<typename M::value_type>)
                        std::copy(std::begin(m[row, column]), std::end(m[row, column]), std::begin((*this)[row, column]));
                    else
                        (*this)[row, column] = static_cast<typename M::value_type>(m[row, column]);
        }

        template <ConceptMatrix M>
        constexpr auto& operator= (const M& m) const {
            assert_extent(m.row_count(), this->row_count(), std::equal_to<>{});
            assert_extent(m.column_count(), this->column_count(), std::equal_to<>{});
            set_from(m);
            return *this;
        }

        constexpr Field& operator[] (IndexType row, IndexType column) const {
            assert(row < this->row_count().get());
            assert(column < this->column_count().get());
            if constexpr (ColumnMajor) {
                return *(base_ptr + column * this->row_count().get() + row);
            } else {
                return *(base_ptr + row * this->column_count().get() + column);
            }
        }

        constexpr auto row_count() const { return rows; }
        constexpr auto column_count() const { return columns; }
    };

    template <typename T, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_ref(T* base_ptr, IndexType rows, IndexType columns) {
        return MatrixReference<T, DynamicExtent, DynamicExtent, ColumnMajor> { base_ptr, rows, columns };
    }

    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_ref(T* base_ptr) {
        return MatrixReference<T, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor> { base_ptr, {}, {} };
    }

    template <typename T, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_ref(const T* base_ptr, IndexType rows, IndexType columns) {
        return MatrixReference<const T, DynamicExtent, DynamicExtent, ColumnMajor> { base_ptr, rows, columns };
    }

    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_ref(const T* base_ptr) {
        return MatrixReference<const T, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor> { base_ptr, {}, {} };
    }

    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_ref(std::initializer_list<T> elements) {
        return mat_ref<Rows, Columns, const T, ColumnMajor>(std::data(elements));
    }

    template <IndexType Rows, IndexType Columns, ConceptContainer T, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_ref(T& container) {
        return mat_ref<Rows, Columns, typename T::value_type, ColumnMajor>(std::data(container));
    }

    template <ConceptContainer T, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_ref(T& container, IndexType rows, IndexType columns) {
        return mat_ref<typename T::value_type, ColumnMajor>(std::data(container), rows, columns);
    }

    template <IndexType Rows, IndexType Columns, ConceptContainer T, bool ColumnMajor = ColumnMajorDefault>
    using mat_ref_static_container_t = decltype(mat_ref<Rows, Columns, T, ColumnMajor>(*static_cast<T*>(0)));

    template <IndexType Rows, IndexType Columns, typename T, bool ColumnMajor = ColumnMajorDefault>
    using mat_ref_static_t = decltype(mat_ref<Rows, Columns, T, ColumnMajor>(std::initializer_list<T>{}));

    template <ConceptContainer T, bool ColumnMajor = ColumnMajorDefault>
    using mat_ref_dynamic_container_t = decltype(mat_ref<T, ColumnMajor>(*static_cast<T*>(0), 0, 0));

    template <typename T, bool ColumnMajor = ColumnMajorDefault>
    using mat_ref_dynamic_t = decltype(mat_ref<T, ColumnMajor>(std::initializer_list<T>{}, 0, 0));

    template <typename T, IndexType Rows, IndexType Columns, bool ColumnMajor>
    struct MatrixObjectStatic {
        std::array<T, Rows * Columns> data;

        using value_type = T;
        constexpr static bool column_major = ColumnMajor;

        constexpr MatrixObjectStatic() = default;
        constexpr MatrixObjectStatic(const MatrixObjectStatic&) = default;
        constexpr MatrixObjectStatic(std::initializer_list<T> elements) {
            std::copy(elements.begin(), elements.end(), data.begin());
        }
        constexpr MatrixObjectStatic(const T& value) {
            std::fill(data.begin(), data.end(), value);
        }

        template <ConceptMatrix M>
        constexpr MatrixObjectStatic(const M& m) { *this = m; }

        template <typename... Values>
        requires (sizeof...(Values) == Rows*Columns)
        constexpr MatrixObjectStatic(Values... values) {
            data = {static_cast<T>(values)...};
        }

        template <typename U>
        using RefType = MatrixReference<U, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor>;

        constexpr auto ref() const { return RefType<const T> { std::data(data) }; }
        auto ref() { return RefType<T> { std::data(data) }; }

        template <ConceptMatrix M>
        constexpr MatrixObjectStatic& operator= (const M& m) { ref() = m; return *this; }

        auto& operator[] (IndexType row, IndexType column) { return ref()[row, column]; }
        const auto& operator[] (IndexType row, IndexType column) const { return ref()[row, column]; }
        auto row_count() const { return ref().row_count(); }
        auto column_count() const { return ref().column_count(); }
    };

    template <typename T, bool ColumnMajor>
    struct MatrixObjectDynamic {
        std::vector<T> data;
        IndexType rows;
        IndexType columns;

        using value_type = T;
        constexpr static bool column_major = ColumnMajor;

        MatrixObjectDynamic()
            : MatrixObjectDynamic(0, 0)
        {}

        MatrixObjectDynamic(const MatrixObjectDynamic&) = default;
        MatrixObjectDynamic(MatrixObjectDynamic&& other) { *this = std::move(other); }

        template <ConceptMatrix M>
        MatrixObjectDynamic(const M& m) : rows{0}, columns{0} { *this = m; }

        MatrixObjectDynamic(IndexType rows, IndexType columns)
            : data(rows * columns)
            , rows { rows }
            , columns { columns }
        {}

        MatrixObjectDynamic(std::initializer_list<T> elements, IndexType rows, IndexType columns)
            : MatrixObjectDynamic(rows, columns)
        {
            std::copy(elements.begin(), elements.end(), data.begin());
        }

        void resize(IndexType rows, IndexType columns) {
            MatrixObjectDynamic other { rows, columns };
            //move-assignment operator is preferred
            //other.ref() = ref();
            other.ref().set_from(ref());
            *this = std::move(other);
        }

        MatrixObjectDynamic& operator= (MatrixObjectDynamic&& other) {
            data = std::move(other.data);
            rows = std::exchange(other.rows, 0);
            columns = std::exchange(other.columns, 0);
            return *this;
        }

        MatrixObjectDynamic& operator= (const MatrixObjectDynamic& other) {
            return (*this).template operator = <MatrixObjectDynamic>(other);
        }

        template <ConceptMatrix M>
        MatrixObjectDynamic& operator= (const M& m) {
            resize(m.row_count().get(), m.column_count().get());
            ref() = m;
            return *this;
        }

        template <typename U>
        using RefType = MatrixReference<U, DynamicExtent, DynamicExtent, ColumnMajor>;

        auto ref() const { return RefType<const T> { std::data(data), rows, columns }; }
        auto ref() { return RefType<T> { std::data(data), rows, columns }; }

        auto& operator[] (IndexType row, IndexType column) { return ref()[row, column]; }
        const auto& operator[] (IndexType row, IndexType column) const { return ref()[row, column]; }
        auto row_count() const { return ref().row_count(); }
        auto column_count() const { return ref().column_count(); }
    };

    template <IndexType Rows, IndexType Columns, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat(std::initializer_list<T> elements) {
        return MatrixObjectStatic<T, Rows, Columns, ColumnMajor> { elements };
    }

    template <IndexType Rows, IndexType Columns, ConceptContainer T, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat(const T& container) {
        return MatrixObjectStatic<T, Rows, Columns, ColumnMajor> { { container } };
    }

    template <typename T, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat(std::initializer_list<T> elements, IndexType rows, IndexType columns) {
        return MatrixObjectDynamic<T, ColumnMajor> { elements, rows, columns };
    }

    template <ConceptContainer T, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat(const T& container, IndexType rows, IndexType columns) {
        return MatrixObjectDynamic<T, ColumnMajor> { container, rows, columns };
    }

    template <ConceptMatrixStatic M, bool ColumnMajor = M::column_major>
    inline auto mat(const M& m) {
        return MatrixObjectStatic<std::remove_const_t<typename M::value_type>, row_count_static<M>(), column_count_static<M>(), ColumnMajor> { m };
    }

    template <ConceptMatrix M, bool ColumnMajor = M::column_major>
    inline auto mat(const M& m) {
        return MatrixObjectDynamic<typename M::value_type, ColumnMajor> { m };
    }

    template <typename T, ConceptMatrixStatic M>
    inline auto mat(const M& m) {
        return MatrixObjectStatic<T, row_count_static<M>(), column_count_static<M>(), M::column_major> { m };
    }

    template <typename T, ConceptMatrix M>
    inline auto mat(const M& m) {
        return MatrixObjectDynamic<T, M::column_major> { m };
    }

    template <typename T, bool ColumnMajor, ConceptMatrixStatic M>
    inline auto mat(const M& m) {
        return MatrixObjectStatic<T, row_count_static<M>(), column_count_static<M>(), ColumnMajor> { m };
    }

    template <typename T, bool ColumnMajor, ConceptMatrix M>
    inline auto mat(const M& m) {
        return MatrixObjectDynamic<T, ColumnMajor> { m };
    }

    template <IndexType Rows, IndexType Columns, typename T, bool ColumnMajor = ColumnMajorDefault>
    using mat_static_t = MatrixObjectStatic<T, Rows, Columns, ColumnMajor>;

    template <typename T, bool ColumnMajor = ColumnMajorDefault>
    using mat_dynamic_t = MatrixObjectDynamic<T, ColumnMajor>;

    template <typename ET>
    constexpr auto invoke_expression_template_result() {
        if constexpr(ConceptMatrix<ET>) {
            if constexpr(ConceptMatrixStatic<ET>)
                return MatrixObjectStatic<typename ET::value_type, row_count_static<ET>(), column_count_static<ET>(), ET::column_major>{};
            else
                return MatrixObjectDynamic<typename ET::value_type, ET::column_major>{};
        } else if constexpr(ConceptVector<ET>) {
            if constexpr(ConceptVectorStatic<ET>)
                return VectorObjectStatic<typename ET::value_type, size_static<ET>()>{};
            else
                return VectorObjectDynamic<typename ET::value_type>{};
        } else return ET{};
    }
    template <typename ET>
    using invoke_expression_template_result_t = decltype(invoke_expression_template_result<ET>());

    template <ConceptExtent E, typename ValueGenerator>
    struct VectorGenerator {
        ValueGenerator f;
        E extent = {};

        using value_type = invoke_expression_template_result_t<decltype(f(0, 0))>;

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            return f(element, extent.get());
        }

        constexpr auto size() const { return extent; }
    };

    template <IndexType Size, typename ValueGenerator>
    constexpr auto vec_procedural(const ValueGenerator& op) {
        return VectorGenerator<StaticExtent<Size>, ValueGenerator> { op };
    }

    template <typename ValueGenerator>
    constexpr auto vec_procedural(IndexType size, const ValueGenerator& op) {
        return VectorGenerator<DynamicExtent, ValueGenerator> { op, size };
    }

    template <ConceptExtent ExtR, ConceptExtent ExtC, typename ValueGenerator, bool ColumnMajor>
    struct MatrixGenerator {
        ValueGenerator f;
        ExtR rows;
        ExtC columns;

        using value_type = invoke_expression_template_result_t<decltype(f(0, 0, 0, 0))>;
        constexpr static bool column_major = ColumnMajor;

        constexpr MatrixGenerator(const ValueGenerator& f = {}, const ExtR& rows = {}, const ExtC& columns = {})
            : f(f), rows(rows), columns(columns)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            return f(row, column, rows.get(), columns.get());
        }

        constexpr auto row_count() const { return rows; }
        constexpr auto column_count() const { return columns; }
    };

    template <IndexType Rows, IndexType Columns, typename ValueGenerator, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_procedural(const ValueGenerator& op) {
        return MatrixGenerator<StaticExtent<Rows>, StaticExtent<Columns>, ValueGenerator, ColumnMajor> { op };
    }
    template <IndexType Rows, IndexType Columns, bool ColumnMajor, typename ValueGenerator>
    constexpr auto mat_procedural(const ValueGenerator& op) {
        return mat_procedural<Rows, Columns, ValueGenerator, ColumnMajor>(op);
    }

    template <typename ValueGenerator, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_procedural(IndexType rows, IndexType columns, const ValueGenerator& op) {
        return MatrixGenerator<DynamicExtent, DynamicExtent, ValueGenerator, ColumnMajor> { op, rows, columns };
    }
    template <bool ColumnMajor, typename ValueGenerator>
    constexpr auto mat_procedural(IndexType rows, IndexType columns, const ValueGenerator& op) {
        return mat_procedural<ValueGenerator, ColumnMajor>(rows, columns, op);
    }

    template <ConceptVector... Vectors>
    struct ColumnVectorMatrix {
        std::array<std::common_type_t<Vectors...>, sizeof...(Vectors)> vectors;

        constexpr static bool column_major = true;

        constexpr ColumnVectorMatrix(Vectors... vs) {
            vectors = {vs...};
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            return vectors[column][row];
        }

        using value_type = decltype(std::declval<ColumnVectorMatrix<Vectors...>>()[0,0]);

        constexpr auto row_count() const { return vectors[0].size(); }
        constexpr auto column_count() const { return StaticExtent<sizeof...(Vectors)>{}; }
    };

    template <ConceptVector... Vectors>
    constexpr auto mat_columns(const Vectors&... vs) {
        return ColumnVectorMatrix<Vectors...>(vs...);
    }

    template <ConceptVector... Vectors>
    constexpr auto mat(Vectors... vs) {
        return mat(mat_columns(vs...));
    }

    template <ConceptMatrix M, ConceptExtent ExtR, ConceptExtent ExtC, typename ValueGenerator>
    struct MatrixExtended {
        M matrix;
        ValueGenerator f;
        ExtR rows;
        ExtC columns;

        using value_type = typename M::value_type;
        constexpr static bool column_major = M::column_major;

        constexpr MatrixExtended(const M& matrix, const ValueGenerator& f = {}, const ExtR& rows = {}, const ExtC& columns = {})
            : matrix(matrix), f(f), rows(rows), columns(columns)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            if(row < matrix.row_count().get() && column < matrix.column_count().get())
                return matrix[row, column];
            return f(row, column, rows.get(), columns.get());
        }

        constexpr auto row_count() const { return rows; }
        constexpr auto column_count() const { return columns; }
    };

    template <IndexType Rows, IndexType Columns, ConceptMatrix M, typename ValueGenerator>
    constexpr auto extend(const M& matrix, const ValueGenerator& op) {
        return MatrixExtended<decltype(matrix.ref()), StaticExtent<Rows>, StaticExtent<Columns>, ValueGenerator> { matrix.ref(), op };
    }
    template <typename ValueGenerator, ConceptMatrix M>
    constexpr auto extend(const M& matrix, IndexType rows, IndexType columns, const ValueGenerator& op) {
        return MatrixExtended<decltype(matrix.ref()), DynamicExtent, DynamicExtent, ValueGenerator> { matrix.ref(), op, rows, columns };
    }

    template <IndexType Rows, IndexType Columns, ConceptMatrix M>
    constexpr auto extend_identity(const M& matrix) {
        return extend<Rows, Columns>(matrix, [](auto i, auto j, auto, auto){ return kronecker_delta<typename M::value_type>(i, j); });
    }
    template <ConceptMatrix M>
    constexpr auto extend_identity(const M& matrix, IndexType rows, IndexType columns) {
        return extend(matrix, rows, columns, [](auto i, auto j, auto, auto){ return kronecker_delta<typename M::value_type>(i, j); });
    }

    template <IndexType Rows, IndexType Columns, ConceptMatrix M>
    constexpr auto extend_constant(const M& matrix, const typename M::value_type& value) {
        return extend<Rows, Columns>(matrix, [&value](auto, auto, auto, auto){ return value; });
    }
    template <ConceptMatrix M>
    constexpr auto extend_constant(const M& matrix, IndexType rows, IndexType columns, const typename M::value_type& value) {
        return extend(matrix, rows, columns, [&value](auto, auto, auto, auto){ return value; });
    }

    template <ConceptMatrix L, ConceptMatrix R>
    //requires (std::same_as<typename L::value_type, typename R::value_type>)
    struct AugmentedMatrix {
        L left;
        R right;

        constexpr static bool column_major = L::column_major;

        constexpr AugmentedMatrix(const L& l, const R& r) : left(l), right(r) {
            assert_extent(left.row_count(), right.row_count(), std::equal_to<>{});
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            if(column >= left.column_count().get())
                return right[row, column - left.column_count().get()];
            return left[row, column];
        }

        using value_type = decltype(std::declval<AugmentedMatrix<L, R>>()[0,0]);

        constexpr auto row_count() const { return left.row_count(); }
        constexpr auto column_count() const { return evaluate_extent(left.column_count(), right.column_count(), std::plus<>{}); }
    };

    template <ConceptMatrix L, ConceptMatrix R> constexpr auto augment(const L& left, const R& right) {
        return AugmentedMatrix { left.ref(), right.ref() };
    }
    template <ConceptMatrix L, ConceptMatrix R> constexpr auto join(const L& left, const R& right) {
        return augment(left, right);
    }

    template <ConceptMatrix M, ConceptExtent E, bool Vertical = false, bool SecondHalf = false>
    struct SplitMatrix {
        M matrix;
        E split_bound;

        using value_type = M::value_type;
        constexpr static bool column_major = M::column_major;

        constexpr SplitMatrix(const M& m, const E& split_bound = {}) : matrix(m), split_bound(split_bound) {
            if constexpr (Vertical) {
                assert_extent(split_bound, matrix.row_count(), std::less<>{});
            } else {
                assert_extent(split_bound, matrix.column_count(), std::less<>{});
            }
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            if constexpr (SecondHalf)
                if constexpr (Vertical) return matrix[row + split_bound.get(), column];
                else return matrix[row, column + split_bound.get()];
            else return matrix[row, column];
        }

        constexpr auto row_count() const {
            if constexpr (Vertical)
                if constexpr (SecondHalf)
                    return evaluate_extent(matrix.row_count(), split_bound, std::minus<>{});
                else
                    return split_bound;
            else return matrix.row_count();
        }
        constexpr auto column_count() const {
            if constexpr (!Vertical)
                if constexpr (SecondHalf)
                    return evaluate_extent(matrix.column_count(), split_bound, std::minus<>{});
                else
                    return split_bound;
            else return matrix.column_count();
        }
    };

    template <ConceptMatrix M> constexpr auto split_left(const M& m, IndexType split_bound) {
        return SplitMatrix<decltype(m.ref()), DynamicExtent, false, false> { m.ref(), split_bound };
    }
    template <IndexType SplitBound, ConceptMatrix M> constexpr auto split_left(const M& m) {
        return SplitMatrix<decltype(m.ref()), StaticExtent<SplitBound>, false, false> { m.ref() };
    }

    template <ConceptMatrix M> constexpr auto split_right(const M& m, IndexType split_bound) {
        return SplitMatrix<decltype(m.ref()), DynamicExtent, false, true> { m.ref(), split_bound };
    }
    template <IndexType SplitBound, ConceptMatrix M> constexpr auto split_right(const M& m) {
        return SplitMatrix<decltype(m.ref()), StaticExtent<SplitBound>, false, true> { m.ref() };
    }

    template <ConceptMatrix M> constexpr auto split_top(const M& m, IndexType split_bound) {
        return SplitMatrix<decltype(m.ref()), DynamicExtent, true, false> { m.ref(), split_bound };
    }
    template <IndexType SplitBound, ConceptMatrix M> constexpr auto split_top(const M& m) {
        return SplitMatrix<decltype(m.ref()), StaticExtent<SplitBound>, true, false> { m.ref() };
    }

    template <ConceptMatrix M> constexpr auto split_bottom(const M& m, IndexType split_bound) {
        return SplitMatrix<decltype(m.ref()), DynamicExtent, true, true> { m.ref(), split_bound };
    }
    template <IndexType SplitBound, ConceptMatrix M> constexpr auto split_bottom(const M& m) {
        return SplitMatrix<decltype(m.ref()), StaticExtent<SplitBound>, true, true> { m.ref() };
    }
    
    template <ConceptMatrix M>
    struct ReducedRowEchelonMatrix {
        using value_type = M::value_type;
        mat_dynamic_t<value_type, M::column_major> matrix;

        constexpr static bool column_major = M::column_major;

        ReducedRowEchelonMatrix(const M& m) : matrix(m.row_count().get(), m.column_count().get()) {
            matrix = m;

            using std::abs;
        
            for(IndexType lead = 0; lead < matrix.row_count().get(); ++lead) {
                value_type divisor, multiplier;
                IndexType pivot = lead;
                for (IndexType row = lead; row < matrix.row_count().get(); ++row)
                    if (abs(matrix[pivot, lead]) < abs(matrix[row, lead]))
                        pivot = row;
                if(pivot != lead)
                    for (IndexType column = 0; column < matrix.column_count().get(); ++column)
                        std::swap(matrix[pivot, column], matrix[lead, column]);

                for (IndexType row = 0; row < matrix.row_count().get(); ++row) {
                    divisor = matrix[lead, lead];
                    if(divisor == static_cast<value_type>(0)) continue;

                    multiplier = matrix[row, lead] / divisor;
                    for (IndexType column = 0; column < matrix.column_count().get(); ++column)
                        if (row == lead)
                            matrix[row, column] /= divisor;
                        else
                            matrix[row, column] -= matrix[lead, column] * multiplier;
                }
            }
        }

        constexpr auto ref() const { return *this; }

        auto operator[] (IndexType row, IndexType column) const {
            return matrix[row, column];
        }
        auto row_count() const { return matrix.row_count(); }
        auto column_count() const { return matrix.column_count(); }
    };

    template <ConceptMatrix M> inline auto reduced_row_echelon_form(const M& m) { return ReducedRowEchelonMatrix{m}; }
    template <ConceptMatrix M> inline auto rref(const M& m) { return reduced_row_echelon_form(m); }

    template <ConceptMatrix M>
    struct Conjugate {
        M matrix;

        constexpr static bool column_major = M::column_major;

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            return conj(matrix[row, column]);
        }

        using value_type = invoke_expression_template_result_t<decltype(std::declval<Conjugate<M>>()[0,0])>;

        constexpr auto row_count() const { return matrix.row_count(); }
        constexpr auto column_count() const { return matrix.column_count(); }
    };

    template <ConceptMatrix M> constexpr auto conjugate(const M& m) { return Conjugate<decltype(m.ref())> { m.ref() }; }
    template <ConceptMatrix M> constexpr auto conj(const M& m) { return conjugate(m); }
    
    template <ConceptMatrix M>
    struct Transpose {
        M matrix;

        using value_type = M::value_type;
        constexpr static bool column_major = M::column_major;

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const { return matrix[column, row]; }
        constexpr auto row_count() const { return matrix.column_count(); }
        constexpr auto column_count() const { return matrix.row_count(); }
    };

    template <ConceptMatrix M> constexpr auto transpose(const M& m) { return Transpose<decltype(m.ref())> { m.ref() }; }

    template <ConceptMatrix M> constexpr auto transpose_hermitian(const M& m) { return transpose(conjugate(m)); }

    template <ConceptMatrix M> constexpr auto gramian(const M& m) { return transpose_hermitian(m)*m; }
    template <ConceptMatrix M> constexpr auto gram(const M& m) { return gramian(m); }
    
    template <ConceptMatrix M, ConceptExtent RowFirst, ConceptExtent Rows, ConceptExtent ColumnFirst, ConceptExtent Columns>
    struct MatrixRectangularPartition {
        M matrix;
        RowFirst row_first;
        Rows rows;
        ColumnFirst column_first;
        Columns columns;

        using value_type = M::value_type;
        constexpr static bool column_major = M::column_major;

        constexpr auto ref() const { return *this; }

        template <ConceptMatrix U>
        MatrixRectangularPartition& operator= (const U& m) {
            for(IndexType row = 0; row < row_count().get(); ++row) {
                for(IndexType column = 0; column < column_count().get(); ++column) {
                    this->operator[](row, column) = m[row, column];
                }
            }
            return *this;
        }

        constexpr auto& operator[] (IndexType row, IndexType column) const {
            return matrix[
                row + row_first.get(),
                column + column_first.get()
            ];
        }

        constexpr auto row_count() const {
            if constexpr(ConceptMatrixStatic<M> && RowFirst::is_static() && Rows::is_static()) {
                if constexpr(row_count_static<M>() < RowFirst::get() + Rows::get()) {
                    return evaluate_extent(matrix.row_count(), row_first, std::minus<>{});
                } else return rows;
            } else {
                if(matrix.row_count().get() < row_first.get() + rows.get())
                    return DynamicExtent { matrix.row_count().get() - row_first.get() };
                else
                    return DynamicExtent { rows.get() };
            }
        }
        constexpr auto column_count() const {
            if constexpr(ConceptMatrixStatic<M> && ColumnFirst::is_static() && Columns::is_static()) {
                if constexpr(column_count_static<M>() < ColumnFirst::get() + Columns::get()) {
                    return evaluate_extent(matrix.column_count(), column_first, std::minus<>{});
                } else return columns;
            } else {
                if(matrix.column_count().get() < column_first.get() + columns.get())
                    return DynamicExtent { matrix.column_count().get() - column_first.get() };
                else
                    return DynamicExtent { columns.get() };
            }
        }
    };

    template <IndexType RowFirst, IndexType Rows, IndexType ColumnFirst, IndexType Columns, ConceptMatrix M>
    constexpr auto rectangular_partition(M& m) {
        return MatrixRectangularPartition<decltype(m.ref()), StaticExtent<RowFirst>, StaticExtent<Rows>, StaticExtent<ColumnFirst>, StaticExtent<Columns>> { m.ref() };
    }

    template <IndexType RowFirst, IndexType Rows, IndexType ColumnFirst, IndexType Columns, ConceptMatrix M>
    constexpr auto rectangular_partition(const M& m) {
        return MatrixRectangularPartition<decltype(m.ref()), StaticExtent<RowFirst>, StaticExtent<Rows>, StaticExtent<ColumnFirst>, StaticExtent<Columns>> { m.ref() };
    }

    template <ConceptMatrix M>
    constexpr auto rectangular_partition(M& m, IndexType row_first, IndexType rows, IndexType column_first, IndexType columns) {
        return MatrixRectangularPartition<decltype(m.ref()), DynamicExtent, DynamicExtent, DynamicExtent, DynamicExtent> {
            m.ref(), row_first, rows, column_first, columns
        };
    }

    template <ConceptMatrix M>
    constexpr auto rectangular_partition(const M& m, IndexType row_first, IndexType rows, IndexType column_first, IndexType columns) {
        return MatrixRectangularPartition<decltype(m.ref()), DynamicExtent, DynamicExtent, DynamicExtent, DynamicExtent> {
            m.ref(), row_first, rows, column_first, columns
        };
    }

    template <ConceptMatrixStatic M, ConceptExtent ErasedRow, ConceptExtent ErasedColumn>
    struct Submatrix {
        M matrix;
        ErasedRow erased_row;
        ErasedColumn erased_column;

        using value_type = M::value_type;
        constexpr static bool column_major = M::column_major;

        constexpr Submatrix(const M& matrix, const ErasedRow& erased_row = {}, const ErasedColumn& erased_column = {})
            : matrix(matrix), erased_row(erased_row), erased_column(erased_column)
        {
            assert_extent(erased_row, matrix.row_count(), std::less<>{});
            assert_extent(erased_column, matrix.column_count(), std::less<>{});
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            row = row >= erased_row.get() ? row+1 : row;
            column = column >= erased_column.get() ? column+1 : column;
            return matrix[row, column];
        }

        constexpr auto row_count() const {
            return evaluate_extent<1>(matrix.row_count(), std::minus<>{});
        }
        constexpr auto column_count() const {
            return evaluate_extent<1>(matrix.column_count(), std::minus<>{});
        }
    };

    template <ConceptMatrix M>
    struct SubmatrixDynamic {
        M matrix;
        std::vector<IndexType> erased_rows;
        std::vector<IndexType> erased_columns;

        using value_type = M::value_type;
        constexpr static bool column_major = M::column_major;

        SubmatrixDynamic(const M& matrix, IndexType erased_row, IndexType erased_column) : matrix(matrix)
        {
            assert(erased_row < matrix.row_count().get());
            assert(erased_column < matrix.column_count().get());
            erased_rows.push_back(erased_row);
            erased_columns.push_back(erased_column);
        }

        SubmatrixDynamic(const SubmatrixDynamic<M>& submatrix, IndexType erased_row, IndexType erased_column)
            : matrix(submatrix.matrix), erased_rows(submatrix.erased_rows), erased_columns(submatrix.erased_columns)
        {
            assert(erased_row < submatrix.row_count().get());
            assert(erased_column < submatrix.column_count().get());
            erased_rows.push_back(erased_row);
            erased_columns.push_back(erased_column);
        }

        constexpr auto ref() const { return *this; }

        auto operator[] (IndexType row, IndexType column) const {
            for(IndexType i = erased_rows.size(); i-- > 0; ) if(erased_rows[i] <= row) ++row;
            for(IndexType i = erased_columns.size(); i-- > 0; ) if(erased_columns[i] <= column) ++column;
            return matrix[row, column];
        }

        auto row_count() const {
            return DynamicExtent(matrix.row_count().get()) - erased_rows.size();
        }
        auto column_count() const {
            return DynamicExtent(matrix.column_count().get()) - erased_columns.size();
        }
    };

    template <IndexType ErasedRow, IndexType ErasedColumn, ConceptMatrixStatic M>
    constexpr auto submatrix(const M& m) {
        return Submatrix<decltype(m.ref()), StaticExtent<ErasedRow>, StaticExtent<ErasedColumn>> { m.ref() };
    }

    template <ConceptMatrix M>
    constexpr auto submatrix(const M& m, IndexType erased_row, IndexType erased_column) {
        return SubmatrixDynamic { m.ref(), erased_row, erased_column };
    }

    template <ConceptMatrix M>
    constexpr auto submatrix(const SubmatrixDynamic<M>& m, IndexType erased_row, IndexType erased_column) {
        return SubmatrixDynamic { m, erased_row, erased_column };
    }

    template <ConceptExtent Rows, ConceptExtent Columns, ConceptMatrix M>
    constexpr auto determinant(const M& m) {
        assert_extent(m.row_count(), m.column_count(), std::equal_to<>{});
        using value_type = typename M::value_type;
        constexpr auto zero = static_cast<value_type>(0);
        if constexpr (Rows::is_static() && Columns::is_static()) {
            if constexpr (Rows::get() == 1) {
                return m[0, 0];
            } else if constexpr (Rows::get() == 2) {
                return m[0, 0]*m[1, 1] - m[0, 1]*m[1, 0];
            } else {
                auto result = zero;
                static_loop<IndexType, Columns::get()>([&result, &m](auto n){
                    result +=
                        m[0, n] * 
                        static_cast<value_type>(n&1?-1:1) * 
                        determinant<StaticExtent<Rows::get()-1>, StaticExtent<Columns::get()-1>>(submatrix<0, n>(m));
                });
                return result;
            }
        } else {
            if(m.row_count().get() == 1) return determinant<StaticExtent<1>, StaticExtent<1>>(m);
            else if(m.row_count().get() == 2) return determinant<StaticExtent<2>, StaticExtent<2>>(m);
            else {
                auto result = zero;
                for (IndexType n = 0; n < m.column_count().get(); n++)
                    result +=
                        m[0, n] * 
                        static_cast<value_type>(n&1?-1:1) * 
                        determinant<DynamicExtent, DynamicExtent>(submatrix(m, 0, n));
                return result;
            }
        }

        return zero;
    }

    template <ConceptMatrixStatic M>
    constexpr auto determinant(const M& m) {
        return determinant<decltype(std::declval<M>().row_count()), decltype(std::declval<M>().column_count())>(m);
    }

    template <ConceptMatrix M>
    constexpr auto determinant(const M& m) {
        return determinant<DynamicExtent, DynamicExtent>(m);
    }
    template <ConceptMatrix M>
    constexpr auto det(const M& m) { return determinant(m); }

    template <ConceptMatrix M>
    struct Cofactor {
        M matrix;

        using value_type = M::value_type;
        constexpr static bool column_major = M::column_major;

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            using value_type = std::remove_reference_t<decltype(matrix[0,0])>;
            return static_cast<value_type>((row+column)&1?-1:1) * determinant(submatrix(matrix, row, column));
        }
        constexpr auto row_count() const { return matrix.row_count(); }
        constexpr auto column_count() const { return matrix.column_count(); }
    };

    template <ConceptMatrix M>
    constexpr auto cofactor(const M& m) { return Cofactor<decltype(m.ref())> { m.ref() }; }
    
    template <ConceptMatrix M>
    constexpr auto adjugate(const M& m) { return transpose(cofactor(m)); }
    
    template <ConceptMatrix M>
    constexpr auto inverse(const M& m) { return adjugate(m)/determinant(m); }
    template <ConceptMatrix M>
    constexpr auto inv(const M& m) { return inverse(m); }

    template <ConceptMatrix M>
    constexpr auto inverse_gauss_jordan(const M& m) {
        auto r = rref(augment(m, mat_identity(m.row_count().get(), m.column_count().get())));
        return split_right(r, m.column_count().get());
    }

    template <ConceptMatrix BasisSource, ConceptMatrix BasisDestination>
    constexpr auto mat_change_of_basis(const BasisSource& basis_src, const BasisDestination& basis_dest) {
        return basis_dest * inverse(basis_src);
    }
    template <ConceptMatrix BasisSource, ConceptMatrix BasisDestination>
    constexpr auto mat_transition(const BasisSource& basis_src, const BasisDestination& basis_dest) {
        return mat_change_of_basis(basis_src, basis_dest);
    }

    template <ConceptMatrix M, ConceptMatrix BasisSource, ConceptMatrix BasisDestination>
    constexpr auto change_of_basis(const M& matrix, const BasisSource& basis_src, const BasisDestination& basis_dest) {
        return mat_change_of_basis(basis_src, basis_dest) * matrix;
    }
    
    template <ConceptMatrix M, ConceptExtent E>
    struct RowOf {
        M matrix;
        E row;

        using value_type = M::value_type;
        constexpr static bool column_major = M::column_major;

        constexpr auto ref() const { return *this; }

        constexpr RowOf(const M& matrix, const E& row = {}) : matrix(matrix), row(row) {
            assert_extent(row, matrix.row_count(), std::less<>{});
        }

        constexpr auto operator[] (IndexType element) const { return matrix[row.get(), element]; }

        constexpr auto size() const { return matrix.column_count(); }
    };

    template <ConceptMatrix M>
    constexpr auto row_of(const M& m, IndexType row) { return RowOf<decltype(m.ref()), DynamicExtent> { m.ref(), row }; }
    
    template <IndexType Row, ConceptMatrix M>
    constexpr auto row_of(const M& m) { return RowOf<decltype(m.ref()), StaticExtent<Row>> { m.ref() }; }

    template <ConceptMatrix M>
    constexpr auto column_of(const M& m, IndexType col) { return row_of(transpose(m), col); }

    template <IndexType Column, ConceptMatrix M>
    constexpr auto column_of(const M& m) { return row_of<Column>(transpose(m)); }

    template <ConceptMatrix M, typename UnaryOperator, bool Indexed>
    struct MatrixUnaryOperation {
        M matrix;
        UnaryOperator op;

        constexpr static bool column_major = M::column_major;

        constexpr MatrixUnaryOperation(const M& m, const UnaryOperator& op = {})
            : matrix(m), op(op)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            if constexpr(Indexed) {
                return op(matrix[row, column], row, column);
            } else {
                return op(matrix[row, column]);
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<MatrixUnaryOperation<M, UnaryOperator, Indexed>>()[0,0])
        >;

        constexpr auto row_count() const { return matrix.row_count(); }
        constexpr auto column_count() const { return matrix.column_count(); }
    };

    template <ConceptMatrix M>
    constexpr auto operator- (const M& m) {
        return MatrixUnaryOperation<decltype(m.ref()), std::negate<>, false> { m.ref() };
    }

    template <bool Indexed, ConceptMatrix M, typename UnaryOperator>
    constexpr auto unary_operation(const M& m, const UnaryOperator& op) {
        return MatrixUnaryOperation<decltype(m.ref()), decltype(op), Indexed> { m.ref(), op };
    }
    template <ConceptMatrix M, typename UnaryOperator, bool Indexed = false>
    constexpr auto unary_operation(const M& m, const UnaryOperator& op) {
        return unary_operation<Indexed, M, UnaryOperator>(m, op);
    }

    template <ConceptMatrix L, ConceptMatrix R>
    struct KroneckerProduct {
        L left;
        R right;

        using value_type = invoke_expression_template_result_t<decltype(left[0,0] * right[0,0])>;
        constexpr static bool column_major = L::column_major;

        constexpr KroneckerProduct(const L& l, const R& r)
            : left(l), right(r)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            return
                left[row/right.row_count().get(), column/right.column_count().get()] *
                right[row%right.row_count().get(), column%right.column_count().get()];
        }

        constexpr auto row_count() const {
            return evaluate_extent(left.row_count(), right.row_count(), std::multiplies<>{});
        }
        constexpr auto column_count() const {
            return evaluate_extent(left.column_count(), right.column_count(), std::multiplies<>{});
        }
    };

    template <ConceptMatrix L, ConceptMatrix R>
    constexpr auto kronecker_product(const L& l, const R& r) {
        return KroneckerProduct { l.ref(), r.ref() };
    };

    template <ConceptMatrix L, ConceptMatrix R>
    struct MatrixMultiplication {
        L left;
        R right;

        using value_type = invoke_expression_template_result_t<decltype(left[0,0]*right[0,0]+left[0,0]*right[0,0])>;
        constexpr static bool column_major = L::column_major;

        constexpr MatrixMultiplication(const L& l, const R& r) : left(l), right(r) {
            assert_extent(left.column_count(), right.row_count(), std::equal_to<>{});
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            auto&& l = row_of(left, row);
            auto&& r = column_of(right, column);
            using T = decltype(l[0]);
            //dot product
            T sum = T{0};
            for (IndexType i = 0; i < l.size().get(); ++i)
                sum += l[i] * r[i];
            return sum;
        }

        constexpr auto row_count() const { return left.row_count(); }
        constexpr auto column_count() const { return right.column_count(); }
    };

    template <ConceptMatrix L, ConceptMatrix R>
    constexpr auto operator* (const L& l, const R& r) {
        return MatrixMultiplication { l.ref(), r.ref() };
    }

    template <ConceptMatrix L, ConceptMatrix R>
    constexpr auto operator/ (const L& l, const R& r) {
        return inverse(r) * l;
    }

    template <ConceptMatrix L, ConceptMatrix R, typename BinaryOperator, bool Indexed>
    struct MatrixComponentWiseBinaryOperation {
        L left;
        R right;
        BinaryOperator op;

        constexpr static bool column_major = L::column_major;

        constexpr MatrixComponentWiseBinaryOperation(const L& l, const R& r, const BinaryOperator& op = {})
            : left(l), right(r), op(op)
        {
            assert_extent(left.row_count(), right.row_count(), std::equal_to<>{});
            assert_extent(left.column_count(), right.column_count(), std::equal_to<>{});
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            if constexpr(Indexed) {
                return op(left[row, column], right[row, column], row, column);
            } else {
                return op(left[row, column], right[row, column]);
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<MatrixComponentWiseBinaryOperation<L, R, BinaryOperator, Indexed>>()[0, 0])
        >;

        constexpr auto row_count() const { return left.row_count(); }
        constexpr auto column_count() const { return left.column_count(); }
    };

    template <bool Indexed, ConceptMatrix L, ConceptMatrix R, typename BinaryOperator>
    constexpr auto binary_operation_component_wise(const L& l, const R& r, const BinaryOperator& op) {
        return MatrixComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), decltype(op), Indexed> { l.ref(), r.ref(), op };
    }
    template <ConceptMatrix L, ConceptMatrix R, typename BinaryOperator, bool Indexed = false>
    constexpr auto binary_operation_component_wise(const L& l, const R& r, const BinaryOperator& op) {
        return binary_operation_component_wise<Indexed, L, R, BinaryOperator>(l, r, op);
    }

    template <ConceptMatrix L, ConceptMatrix R>
    constexpr auto operator+ (const L& l, const R& r) {
        return MatrixComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), std::plus<>, false> { l.ref(), r.ref() };
    }

    template <ConceptMatrix L, ConceptMatrix R>
    constexpr auto operator- (const L& l, const R& r) {
        return MatrixComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), std::minus<>, false> { l.ref(), r.ref() };
    }

    template <ConceptMatrix L, ConceptMatrix R>
    constexpr auto hadamard_product(const L& l, const R& r) {
        return MatrixComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), std::multiplies<>, false> { l.ref(), r.ref() };
    }

    template <ConceptMatrix L, ConceptMatrix R>
    constexpr auto hadamard_division(const L& l, const R& r) {
        return MatrixComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), std::divides<>, false> { l.ref(), r.ref() };
    }

    template <ConceptMatrix L, typename Field, typename BinaryOperator, bool Indexed>
    struct MatrixScalarBinaryOperation {
        L left;
        Field right;
        BinaryOperator op;

        constexpr static bool column_major = L::column_major;

        constexpr MatrixScalarBinaryOperation(const L& l, const Field& r, const BinaryOperator& op = {})
            : left(l), right(r), op(op)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            if constexpr(Indexed) {
                return op(left[row, column], right, row, column);
            } else {
                return op(left[row, column], right);
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<MatrixScalarBinaryOperation<L, Field, BinaryOperator, Indexed>>()[0,0])
        >;

        constexpr auto row_count() const { return left.row_count(); }
        constexpr auto column_count() const { return left.column_count(); }
    };

    template <ConceptMatrix L, typename Field>
    constexpr auto operator* (const L& l, const Field& r) {
        return MatrixScalarBinaryOperation<decltype(l.ref()), Field, std::multiplies<>, false> { l.ref(), r };
    }
    template <typename Field, ConceptMatrix R>
    constexpr auto operator* (const Field& l, const R& r) { return r * l; }

    template <ConceptMatrix L, typename Field>
    constexpr auto operator/ (const L& l, const Field& r) {
        return MatrixScalarBinaryOperation<decltype(l.ref()), Field, std::divides<>, false> { l.ref(), r };
    }

    template <ConceptMatrix L, typename Field>
    constexpr auto operator+ (const L& l, const Field& r) {
        return MatrixScalarBinaryOperation<decltype(l.ref()), Field, std::plus<>, false> { l.ref(), r };
    }
    template <typename Field, ConceptMatrix R>
    constexpr auto operator+ (const Field& l, const R& r) { return r + l; }

    template <ConceptMatrix L, typename Field>
    constexpr auto operator- (const L& l, const Field& r) {
        return MatrixScalarBinaryOperation<decltype(l.ref()), Field, std::minus<>, false> { l.ref(), r };
    }
    
    template <ConceptMatrix L, typename R>
    constexpr auto& operator*= (L& l, const R& r) { l = r * l; return l; }
    template <ConceptMatrix L, typename R>
    constexpr auto& operator/= (L& l, const R& r) { l = l / r; return l; }
    template <ConceptMatrix L, typename R>
    constexpr auto& operator+= (L& l, const R& r) { l = l + r; return l; }
    template <ConceptMatrix L, typename R>
    constexpr auto& operator-= (L& l, const R& r) { l = l - r; return l; }

    template <bool Indexed, ConceptMatrix A, typename B, typename BinaryOperator>
    constexpr auto binary_operation(const A& m, const B& b, const BinaryOperator& op) {
        return MatrixScalarBinaryOperation<decltype(m.ref()), B, decltype(op), Indexed> { m.ref(), b, op };
    }
    template <ConceptMatrix A, typename B, typename BinaryOperator, bool Indexed = false>
    constexpr auto binary_operation(const A& m, const B& b, const BinaryOperator& op) {
        return binary_operation<Indexed, A, B, BinaryOperator>(m, b, op);
    }

    template <ConceptMatrix A, typename B, typename C, typename TernaryOperator, bool Indexed>
    struct MatrixTernaryOperation {
        A matrix;
        B b;
        C c;
        TernaryOperator op;

        constexpr static bool column_major = A::column_major;

        constexpr MatrixTernaryOperation(const A& a, const B& b, const C& c, const TernaryOperator& op = {})
            : matrix(a), b(b), c(c), op(op)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            if constexpr(Indexed) {
                return op(matrix[row, column], b, c, row, column);
            } else {
                return op(matrix[row, column], b, c);
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<MatrixTernaryOperation<A, B, C, TernaryOperator, Indexed>>()[0,0])
        >;

        constexpr auto row_count() const { return matrix.row_count(); }
        constexpr auto column_count() const { return matrix.column_count(); }
    };

    template <ConceptMatrix A, typename B, typename C>
    constexpr auto clamp(const A& m, const B& lower, const C& upper) {
        auto op = [](const auto& x, const auto& lower, const auto& upper)->auto {
            using std::clamp;
            using Maths::clamp;
            return clamp(x, lower, upper);
        };
        return MatrixTernaryOperation<decltype(m.ref()), B, C, decltype(op), false> { m.ref(), lower, upper, {} };
    }

    template <bool Indexed, ConceptMatrix A, typename B, typename C, typename TernaryOperator>
    constexpr auto ternary_operation(const A& a, const B& b, const C& c, const TernaryOperator& op) {
        return MatrixTernaryOperation<decltype(a.ref()), B, C, TernaryOperator, Indexed> { a.ref(), b, c, op };
    }
    template <ConceptMatrix A, typename B, typename C, typename TernaryOperator, bool Indexed = false>
    constexpr auto ternary_operation(const A& a, const B& b, const C& c, const TernaryOperator& op) {
        return ternary_operation<Indexed, A, B, C, TernaryOperator>(a, b, c, op);
    }

    template <ConceptMatrix M>
    constexpr auto trace(const M& m) {
        assert_extent(m.row_count(), m.column_count(), std::equal_to<>{});
        auto sum = m[0,0];
        for(IndexType i = 1; i < m.row_count().get(); ++i)
            sum += m[i, i];
        return sum;
    }
    template <ConceptMatrix M>
    constexpr auto tr(const M& m) { return trace(m); }

    template <ConceptMatrix M>
    constexpr auto min(const M& m) {
        auto minimum = m[0,0];
        for(IndexType row = 0; row < m.row_count().get(); ++row)
            for(IndexType column = 0; column < m.column_count().get(); ++column)
                if(minimum > m[row, column]) minimum = m[row, column];
        return minimum;
    }

    template <ConceptMatrix M>
    constexpr auto max(const M& m) {
        auto maximum = m[0,0];
        for(IndexType row = 0; row < m.row_count().get(); ++row)
            for(IndexType column = 0; column < m.column_count().get(); ++column)
                if(maximum < m[row, column]) maximum = m[row, column];
        return maximum;
    }

    template <ConceptMatrix M>
    constexpr auto norm_max(const M& m) {
        using std::abs;
        auto norm = abs(m[0,0]);
        for(IndexType row = 0; row < m.row_count().get(); ++row)
            for(IndexType column = 0; column < m.column_count().get(); ++column)
                if(norm < abs(m[row, column])) norm = abs(m[row, column]);
        return norm;
    }

    template <ConceptMatrix M>
    constexpr auto norm_frobenius(const M& m) {
        using std::sqrt;
        auto sum = static_cast<typename M::value_type>(0);
        for(IndexType row = 0; row < m.row_count().get(); ++row)
            for(IndexType column = 0; column < m.column_count().get(); ++column)
                sum += m[row, column] * m[row, column];
        return sqrt(sum);
    }
    template <ConceptMatrix M>
    constexpr auto norm_euclidean(const M& m) { return norm_frobenius(m); }
    template <ConceptMatrix M>
    constexpr auto norm(const M& m) { return norm_frobenius(m); }

    template <ConceptMatrix M>
    constexpr auto normalize_frobenius(const M& m) {
        return m/norm_frobenius(m);
    }
    template <ConceptMatrix M>
    constexpr auto normalize_euclidean(const M& m) { return normalize_frobenius(m); }
    template <ConceptMatrix M>
    constexpr auto normalize(const M& m) { return normalize_frobenius(m); }

    template <ConceptMatrix M>
    constexpr auto normalize_max(const M& m) { return m/typename M::value_type{norm_max(m)}; }

    template <ConceptMatrix M>
    constexpr auto normalize_minmax(const M& m) {
        auto minimum = min(m);
        auto maximum = max(m);
        return (m - minimum)/(maximum - minimum);
    }

    template <typename T, ConceptExtent ExtD, bool ColumnMajor>
    struct DiscreteFourierTransformMatrix {
        using value_type = std::complex<T>;
        constexpr static bool column_major = ColumnMajor;

        ExtD dimension;
        value_type omega;
        T norm;

        constexpr DiscreteFourierTransformMatrix(const ExtD& dimension = {})
            : dimension(dimension)
        {
            constexpr value_type i = value_type(static_cast<T>(0), static_cast<T>(1));
            constexpr T pi = std::numbers::pi_v<T>;
            
            omega = std::exp(static_cast<T>(-2) * pi * i / value_type(static_cast<T>(dimension.get())));
            norm = static_cast<T>(1)/std::sqrt(static_cast<T>(dimension.get()));
        }

        constexpr auto ref() const { return *this; }

        constexpr value_type operator[] ([[maybe_unused]] IndexType row, [[maybe_unused]] IndexType column) const {
            return std::pow(omega, static_cast<T>(column*row))*norm;
        }

        constexpr auto row_count() const { return dimension; }
        constexpr auto column_count() const { return dimension; }
    };

    template <IndexType Dimension, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_DFT() {
        return DiscreteFourierTransformMatrix<T, StaticExtent<Dimension>, ColumnMajor> {};
    }
    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_DFT(IndexType dimension) {
        return DiscreteFourierTransformMatrix<T, DynamicExtent, ColumnMajor> { dimension };
    }

    template <ConceptExtent Dimension, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_walsh_sylvester([[maybe_unused]]IndexType dimension) {
        if constexpr (Dimension::is_static()) {
            if constexpr (Dimension::get() == 1) {
                //Hadamard matrix of order 1
                return mat<1, 1, T, ColumnMajor>({1});
            } else if constexpr (Dimension::get() == 2) {
                //Hadamard matrix of order 2
                return mat<2, 2, T, ColumnMajor>({
                    1,  1,
                    1, -1,
                });
            } else {
                //Hadamard matrix of order N
                auto H_2 = mat_walsh_sylvester<StaticExtent<2>, T, ColumnMajor>(0);
                auto H_n = mat_walsh_sylvester<StaticExtent<Dimension::get()/2>, T, ColumnMajor>(0);
                return mat(kronecker_product(H_2, H_n));
            }
        } else {
            if (dimension==1) {
                return mat<T, ColumnMajor>({1}, 1, 1);
            } else if (dimension==2) {
                return mat<T, ColumnMajor>({ 1,  1, 1, -1, }, 2, 2);
            } else {
                auto H_2 = mat_walsh_sylvester<DynamicExtent, T, ColumnMajor>(2);
                auto H_n = mat_walsh_sylvester<DynamicExtent, T, ColumnMajor>(dimension/2);
                mat_dynamic_t<T, ColumnMajor> ret = kronecker_product(H_2, H_n);
                return ret;
            }
        }
    }

    template <IndexType Dimension, typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto mat_walsh_sylvester() {
        static_assert(is_power_of_two(Dimension));
        return mat_walsh_sylvester<StaticExtent<Dimension>, T, ColumnMajor>(0);
    }

    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat_walsh_sylvester(IndexType dimension) {
        assert(is_power_of_two(dimension));
        return mat_walsh_sylvester<DynamicExtent, T, ColumnMajor>(dimension);
    }

    template <typename T, bool ColumnMajor>
    struct MatrixPerspectiveProjection {
        T left, right, bottom, top, near, far, depth_near, depth_far;

        using value_type = T;
        constexpr static bool column_major = ColumnMajor;

        MatrixPerspectiveProjection(
            const T& left, const T& right,
            const T& bottom, const T& top,
            const T& near, const T& far,
            const T& depth_near, const T& depth_far
        ) : 
        left(left), right(right), bottom(bottom), top(top),
        near(near), far(far), depth_near(depth_near), depth_far(depth_far)
        {}

        constexpr auto ref() const { return *this; }

        auto operator[] (IndexType row, IndexType column) const {
            using std::numeric_limits;
            constexpr value_type zero{0};
            constexpr value_type one{1};
            constexpr value_type two{2};
            switch(column) {
                case 0: {
                    switch(row) {
                        case 0: return two*near/(right - left);
                        default: return zero;
                    }
                }
                case 1: {
                    switch(row) {
                        case 1: return two*near/(top - bottom);
                        default: return zero;
                    }
                }
                case 2: {
                    switch(row) {
                        case 0: return (right + left)/(right - left);
                        case 1: return (top + bottom)/(top - bottom);
                        case 2: return
                            far != numeric_limits<decltype(far)>::infinity()?
                            (depth_far*far - depth_near*near)/(far - near) :
                            depth_far;
                        case 3: return one;
                        default: return zero;
                    }
                }
                case 3: {
                    switch(row) {
                        case 2: return
                            far != numeric_limits<decltype(far)>::infinity()?
                            -(far*near*(depth_far - depth_near)/(far - near)) :
                            -near*(depth_far - depth_near);
                        default: return zero;
                    }
                }
                default: return kronecker_delta<value_type>(row, column);
            }
        }

        auto row_count() const { return StaticExtent<4>{}; }
        auto column_count() const { return StaticExtent<4>{}; }
    };

    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat_projection_perspective(
        const T& left, const T& right,
        const T& bottom, const T& top,
        const T& near, const T& far,
        const T& depth_near, const T& depth_far) {
        return MatrixPerspectiveProjection<T, ColumnMajor>(left, right, bottom, top, near, far, depth_near, depth_far);
    }

    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat_projection_perspective(
        const T& field_of_view,
        const T& aspect_ratio,
        const T& near, const T& far,
        const T& depth_near, const T& depth_far) {
        using std::tan;
        T tangent = tan(field_of_view/2);
        T top = near * tangent;
        T right = top * aspect_ratio;
        return MatrixPerspectiveProjection<T, ColumnMajor>(-right, right, -top, top, near, far, depth_near, depth_far);
    }

    template <typename T, bool ColumnMajor>
    struct MatrixOrthographicProjection {
        T left, right, bottom, top, near, far, depth_near, depth_far;

        using value_type = T;
        constexpr static bool column_major = ColumnMajor;

        MatrixOrthographicProjection(
            const T& left, const T& right,
            const T& bottom, const T& top,
            const T& near, const T& far,
            const T& depth_near, const T& depth_far
        ) : 
        left(left), right(right), bottom(bottom), top(top),
        near(near), far(far), depth_near(depth_near), depth_far(depth_far)
        {}

        constexpr auto ref() const { return *this; }

        auto operator[] (IndexType row, IndexType column) const {
            using std::numeric_limits;
            constexpr value_type zero{0};
            constexpr value_type one{1};
            constexpr value_type two{2};
            switch(column) {
                case 0: {
                    switch(row) {
                        case 0: return two/(right - left);
                        default: return zero;
                    }
                }
                case 1: {
                    switch(row) {
                        case 1: return two/(top - bottom);
                        default: return zero;
                    }
                }
                case 2: {
                    switch(row) {
                        case 2: return (depth_far - depth_near)/(far - near);
                        default: return zero;
                    }
                }
                case 3: {
                    switch(row) {
                        case 0: return -(right + left)/(right - left);
                        case 1: return -(top + bottom)/(top - bottom);
                        case 2: return (depth_near*far - depth_far*near)/(far - near);
                        case 3: return one;
                        default: return zero;
                    }
                }
                default: return kronecker_delta<value_type>(row, column);
            }
        }

        auto row_count() const { return StaticExtent<4>{}; }
        auto column_count() const { return StaticExtent<4>{}; }
    };

    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat_projection_orthographic(
        const T& left, const T& right,
        const T& bottom, const T& top,
        const T& near, const T& far,
        const T& depth_near, const T& depth_far) {
        return MatrixOrthographicProjection<T, ColumnMajor>(left, right, bottom, top, near, far, depth_near, depth_far);
    }

    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat_projection_orthographic(
        const T& width,
        const T& height,
        const T& near, const T& far,
        const T& depth_near, const T& depth_far) {
        T top = width/2;
        T right = height/2;
        return MatrixOrthographicProjection<T, ColumnMajor>(-right, right, -top, top, near, far, depth_near, depth_far);
    }

    template <typename T = NumericalTypeDefault, bool ColumnMajor = ColumnMajorDefault>
    inline auto mat_projection_orthographic(
        const T& width,
        const T& height,
        const T& depth,
        const T& depth_near, const T& depth_far) {
        T top = width/2;
        T right = height/2;
        T far = height/2;
        return MatrixOrthographicProjection<T, ColumnMajor>(-right, right, -top, top, -far, far, depth_near, depth_far);
    }

    template <ConceptVector V, bool ColumnMajor>
    struct MatrixScaling {
        V coefficients;

        using value_type = V::value_type;
        constexpr static bool column_major = ColumnMajor;

        constexpr MatrixScaling(const V& coeffs) : coefficients(coeffs) {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            return row == column ?
                coefficients[row] :
                static_cast<decltype(coefficients[row])>(0);
        }

        constexpr auto row_count() const { return coefficients.size(); }
        constexpr auto column_count() const { return coefficients.size(); }
    };

    template <bool ColumnMajor, ConceptVector V>
    constexpr auto scaling(const V& coefficients) {
        return MatrixScaling<decltype(coefficients.ref()), ColumnMajor> { coefficients.ref() };
    }
    template <ConceptVector V, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto scaling(const V& coefficients) { return scaling<ColumnMajor>(coefficients); }

    template <ConceptVector V, bool ColumnMajor>
    struct MatrixTranslation {
        V coefficients;

        using value_type = V::value_type;
        constexpr static bool column_major = ColumnMajor;

        constexpr MatrixTranslation(const V& coeffs) : coefficients(coeffs) {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            constexpr auto zero = static_cast<decltype(coefficients[row])>(0);
            constexpr auto one = static_cast<decltype(coefficients[row])>(1);
            return row == column ? one : (column == coefficients.size().get()? coefficients[row] : zero);
        }

        constexpr auto row_count() const { return evaluate_extent<1>(coefficients.size(), std::plus<>{}); }
        constexpr auto column_count() const { return evaluate_extent<1>(coefficients.size(), std::plus<>{}); }
    };

    template <bool ColumnMajor, ConceptVector V>
    constexpr auto translation(const V& coefficients) {
        return MatrixTranslation<decltype(coefficients.ref()), ColumnMajor> { coefficients.ref() };
    }
    template <ConceptVector V, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto translation(const V& coefficients) { return translation<ColumnMajor>(coefficients); }

    template <typename Field, ConceptVector U, ConceptVector V, bool ColumnMajor>
    struct MatrixRotation {
        U basis_u;
        V basis_v;
        Field theta;

        using value_type = V::value_type;
        constexpr static bool column_major = ColumnMajor;

        MatrixRotation(const V& basis_u, const V& basis_v, Field theta)
            : basis_u(basis_u), basis_v(basis_v), theta(theta)
        {
            assert_extent(basis_u.size(), basis_v.size(), std::equal_to<>{});
        }

        constexpr auto ref() const { return *this; }

        auto operator[] (IndexType row, IndexType column) const {
            //A=I+sin(θ)(vu^T−uv^T)+(cos(θ)−1)(uu^T+vv^T)
            return
                kronecker_delta<Field>(row, column) + 
                std::sin(theta) * 
                (as_column(basis_v)*as_row(basis_u) - as_column(basis_u)*as_row(basis_v))[row, column] + 
                (std::cos(theta) - static_cast<Field>(1)) * 
                (as_column(basis_u)*as_row(basis_u) + as_column(basis_v)*as_row(basis_v))[row, column];
        }

        auto row_count() const { return basis_u.size(); }
        auto column_count() const { return basis_u.size(); }
    };

    template <bool ColumnMajor, typename Field, ConceptVector U, ConceptVector V>
    inline auto rotation(const U& basis_u, const V& basis_v, Field theta) {
        return MatrixRotation<Field, decltype(basis_u.ref()), decltype(basis_v.ref()), ColumnMajor> {
            basis_u.ref(), basis_v.ref(), theta
        };
    }
    template <typename Field, ConceptVector U, ConceptVector V, bool ColumnMajor = ColumnMajorDefault>
    inline auto rotation(const U& basis_u, const V& basis_v, Field theta) {
        return rotation<ColumnMajor>(basis_u, basis_v, theta);
    }

    template <ConceptVector V, bool ColumnMajor>
    struct AsRowVector {
        V vector;

        using value_type = V::value_type;
        constexpr static bool column_major = ColumnMajor;

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            assert(row == 0);
            return vector[column];
        }

        constexpr auto row_count() const { return StaticExtent<1>(); }
        constexpr auto column_count() const { return vector.size(); }
    };

    template <bool ColumnMajor, ConceptVector V>
    constexpr auto as_row(const V& vector) {
        return AsRowVector<decltype(vector.ref()), ColumnMajor>{ vector.ref() };
    }
    template <ConceptVector V, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_row(const V& vector) {
        return as_row<ColumnMajor>(vector);
    }

    template <ConceptVector V, bool ColumnMajor>
    struct AsColumnVector {
        V vector;

        using value_type = V::value_type;
        constexpr static bool column_major = ColumnMajor;

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            assert(column == 0);
            return vector[row];
        }

        constexpr auto row_count() const { return vector.size(); }
        constexpr auto column_count() const { return StaticExtent<1>(); }
    };

    template <bool ColumnMajor, ConceptVector V>
    constexpr auto as_column(const V& vector) {
        return AsColumnVector<decltype(vector.ref()), ColumnMajor>{ vector.ref() };
    }
    template <ConceptVector V, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_column(const V& vector) {
        return as_column<ColumnMajor>(vector);
    }

    template <ConceptVector V, ConceptExtent E, bool ColumnMajor>
    struct VectorAsMatrix {
        V vector;
        E stride;

        using value_type = V::value_type;
        constexpr static bool column_major = ColumnMajor;

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            if constexpr (ColumnMajor) {
                return vector[column*stride.get() + row];
            } else {
                return vector[row*stride.get() + column];
            }
        }

        constexpr auto row_count() const {
            if constexpr (!ColumnMajor)
                return evaluate_extent(vector.size(), stride, std::divides<>{});
            else return stride;
        }
        constexpr auto column_count() const {
            if constexpr (ColumnMajor)
                return evaluate_extent(vector.size(), stride, std::divides<>{});
            else return stride;
        }
    };

    template <IndexType Stride, bool ColumnMajor, ConceptVector V>
    constexpr auto as_matrix(const V& vector) {
        return VectorAsMatrix<decltype(vector.ref()), StaticExtent<Stride>, ColumnMajor>{ vector.ref(), {} };
    }
    template <IndexType Stride, ConceptVector V, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_matrix(const V& vector) {
        return as_matrix<Stride, ColumnMajor>(vector);
    }

    template <bool ColumnMajor, ConceptVector V>
    constexpr auto as_matrix(const V& vector, IndexType stride) {
        return VectorAsMatrix<decltype(vector.ref()), DynamicExtent, ColumnMajor>{ vector.ref(), stride };
    }
    template <ConceptVector V, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_matrix(const V& vector, IndexType stride) {
        return as_matrix<ColumnMajor>(vector, stride);
    }

    template <ConceptVector V, typename UnaryOperator, bool Indexed>
    struct VectorUnaryOperation {
        V vector;
        UnaryOperator op;

        constexpr VectorUnaryOperation(const V& v, const UnaryOperator& op = {})
            : vector(v), op(op)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            if constexpr(Indexed) {
                return op(vector[element], element);
            } else {
                return op(vector[element]);
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<VectorUnaryOperation<V, UnaryOperator, Indexed>>()[0])
        >;

        constexpr auto size() const { return vector.size(); }
    };

    template <ConceptVector V>
    constexpr auto operator- (const V& v) {
        return VectorUnaryOperation<decltype(v.ref()), std::negate<>, false> { v.ref() };
    }

    template <bool Indexed, ConceptVector V, typename UnaryOperator>
    constexpr auto unary_operation(const V& v, const UnaryOperator& op) {
        return VectorUnaryOperation<decltype(v.ref()), decltype(op), Indexed> { v.ref(), op };
    }
    template <ConceptVector V, typename UnaryOperator, bool Indexed = false>
    constexpr auto unary_operation(const V& v, const UnaryOperator& op) {
        return unary_operation<Indexed, V, UnaryOperator>(v, op);
    }

    template<ConceptVector A>
    constexpr auto abs(const A& a) {
        return unary_operation(a, [](auto x){ using std::abs; return abs(x); });
    }

    template<ConceptVector A>
    constexpr auto sign(const A& a) {
        return unary_operation(a, [](auto x){ return std::signbit(x) ? -1 : 1; });
    }

    template <ConceptVector A, ConceptVector B>
    requires is_complex_v<typename B::value_type>
    constexpr auto inner_product(const A& a, const B& b) {
        auto ret = (as_row(a) * conj(as_column(b)))[0, 0];
        return std::real(ret);
    }
    template <ConceptVector A, ConceptVector B>
    constexpr auto inner_product_euclidean(const A& a, const B& b) {
        using T = decltype(a[0]*b[0] + a[0]*b[0]);
        static constexpr size_t N = 4;
        const size_t count = a.size().get();

        size_t i = 0;
        T sum = T{0};
        if (count >= N) {
            T sum_vec[N];
            for (size_t j = 0; j < N; ++j)
                sum_vec[j] = T{0};

            for (; i + N < count; i += N) {
                for (size_t j = 0; j < N; ++j)
                    sum_vec[j] += a[i + j] * b[i + j];
            }
            
            sum = sum_vec[0];
            for (size_t j = 1; j < N; ++j)
                sum += sum_vec[j];
        }

        for (; i < count; ++i)
            sum += a[i] * b[i];
        return sum;
    }
    template <ConceptVector A, ConceptVector B>
    requires
        (std::is_floating_point_v<typename A::value_type> && std::is_floating_point_v<typename B::value_type>) ||
        (std::is_integral_v<typename A::value_type> && std::is_integral_v<typename B::value_type>)
    constexpr auto inner_product(const A& a, const B& b) { return inner_product_euclidean(a, b); }
    template <ConceptVector A, ConceptVector B>
    constexpr auto dot_product(const A& a, const B& b) { return inner_product_euclidean(a, b); }
    template <ConceptVector A, ConceptVector B>
    constexpr auto dot(const A& a, const B& b) { return inner_product_euclidean(a, b); }

    template <ConceptVector L, ConceptVector R, bool ColumnMajor>
    struct OuterProduct {
        L left;
        R right;

        using value_type = invoke_expression_template_result_t<decltype(left[0] * right[0])>;
        constexpr static bool column_major = ColumnMajor;

        constexpr OuterProduct(const L& l, const R& r)
            : left(l), right(r)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            return left[row]*right[column];
        }

        constexpr auto row_count() const { return left.size(); }
        constexpr auto column_count() const { return right.size(); }
    };

    template <bool ColumnMajor, ConceptVector L, ConceptVector R>
    constexpr auto outer_product(const L& l, const R& r) {
        return OuterProduct<decltype(l.ref()), decltype(r.ref()), ColumnMajor>{ l.ref(), r.ref() };
    }
    template <ConceptVector L, ConceptVector R, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto outer_product(const L& l, const R& r) {
        return outer_product<ColumnMajor>(l, r);
    }

    template <ConceptVector L, ConceptVector R>
    struct CrossProduct {
        L left;
        R right;

        using value_type = invoke_expression_template_result_t<decltype(left[0]*right[0] - left[0]*right[0])>;

        constexpr CrossProduct(const L& l, const R& r)
            : left(l), right(r)
        {
            assert_extent(left.size(), StaticExtent<3>{}, std::equal_to<>{});
            assert_extent(right.size(), StaticExtent<3>{}, std::equal_to<>{});
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType i) const {
            switch(i) {
                case 0: return left[1]*right[2] - left[2]*right[1];
                case 1: return left[2]*right[0] - left[0]*right[2];
                case 2: return left[0]*right[1] - left[1]*right[0];
            }
            return static_cast<value_type>(0);
        }

        constexpr auto size() const { return StaticExtent<3>{}; }
    };

    template <ConceptVector L, ConceptVector R>
    constexpr auto cross_product(const L& l, const R& r) {
        return CrossProduct{ l.ref(), r.ref() };
    }
    template <ConceptVector L, ConceptVector R>
    constexpr auto cross(const L& l, const R& r) { return cross_product( l, r ); }
    
    template <ConceptVector L, ConceptVector R>
    constexpr auto kronecker_product(const L& l, const R& r) {
        return column_of(kronecker_product( as_column(l), as_column(r) ), 0);
    }
    template <ConceptVectorStatic L, ConceptVectorStatic R>
    constexpr auto kronecker_product(const L& l, const R& r) {
        return column_of<0>(kronecker_product( as_column(l), as_column(r) ));
    }

    template <ConceptMatrix M, ConceptVectorStatic V>
    constexpr auto operator* (const M& m, const V& v) { return column_of<0>(m.ref() * as_column(v)); }
    template <ConceptVectorStatic V, ConceptMatrix M>
    constexpr auto operator* (const V& v, const M& m) { return row_of<0>(as_row(v) * m.ref()); }

    template <ConceptMatrix M, ConceptVector V>
    constexpr auto operator* (const M& m, const V& v) { return column_of(m.ref() * as_column(v), 0); }
    template <ConceptVector V, ConceptMatrix M>
    constexpr auto operator* (const V& v, const M& m) { return row_of(as_row(v) * m.ref(), 0); }

    template <ConceptVector V, ConceptVector N, typename T>
    constexpr bool is_total_internal_reflection(const V& v, const N& n, T eta) {
        assert_extent(v.size(), n.size(), std::equal_to<>{});
        auto zero = static_cast<T>(0);
        auto one = static_cast<T>(1);
        auto nv = inner_product(n, v);
        auto k = one - eta * eta * (one - nv * nv);
        return k < zero;
    }
    template <ConceptVector V, ConceptVector N, typename T>
    constexpr bool is_TIR(const V& v, const N& n, T eta) {
        return is_total_internal_reflection(v, n, eta);
    }

    template <Conventions::RayDirection RayDirectionConvention, ConceptVector V, ConceptVector N>
    constexpr auto reflect(const V& v, const N& n) {
        using value_type = decltype(inner_product(v, n));
        assert_extent(v.size(), n.size(), std::equal_to<>{});
        if constexpr (RayDirectionConvention == Conventions::RayDirection::Incoming)
            return v - static_cast<value_type>(2)*inner_product(v, n)*n;
        else
            return -v - static_cast<value_type>(2)*-inner_product(v, n)*n;
    }
    template <
    ConceptVector V,
    ConceptVector N,
    Conventions::RayDirection RayDirectionConvention = RayDirectionDefault>
    constexpr auto reflect(const V& v, const N& n) {
        return reflect<RayDirectionConvention>(v, n);
    }

    template <ConceptVector V, ConceptVector N, typename T, Conventions::RayDirection RayDirectionConvention, bool TIR>
    struct Refract {
        V vector;
        N normal;
        T eta;
        decltype(inner_product(normal, vector)) nv;
        decltype(T{1} - eta * eta * (T{1} - nv * nv)) k;

        constexpr Refract(const V& v, const N& n, T eta)
            : vector(v), normal(n), eta(eta)
        {
            assert_extent(v.size(), n.size(), std::equal_to<>{});
            T one {1};
            nv = inner_product(n, v);
            if constexpr (RayDirectionConvention == Conventions::RayDirection::Outgoing)
                this->eta = -eta;
            k = one - this->eta * this->eta * (one - nv * nv);
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType i) const {
            using std::sqrt;
            using value_type = decltype(eta * vector[i] - (eta * nv + sqrt(k)) * normal[i]);
            decltype(k) zero {0};
            if constexpr(TIR) {
                // automatic reflection upon satisfaction of total internal reflection condition
                if (k < zero) {
                    if constexpr (RayDirectionConvention == Conventions::RayDirection::Incoming)
                        return vector[i] - value_type{2}*nv*normal[i];
                    else
                        return -vector[i] - value_type{2}*-nv*normal[i];
                }
            } else {
                // GLSL behaviour
                if (k < zero) return value_type{0};
            }
            return eta * vector[i] - (eta * nv + sqrt(k)) * normal[i];
        }

        using value_type = invoke_expression_template_result_t<decltype(std::declval<Refract<V,N,T,RayDirectionConvention,TIR>>()[0])>;

        constexpr auto size() const { return vector.size(); }
    };

    template <Conventions::RayDirection RayDirectionConvention, bool TIR, ConceptVector V, ConceptVector N, typename T>
    constexpr auto refract(const V& v, const N& n, T eta) {
        return Refract<decltype(v.ref()), decltype(n.ref()), T, RayDirectionConvention, TIR> { v.ref(), n.ref(), eta };
    }
    template <
    ConceptVector V, ConceptVector N, typename T,
    bool TIR = true,
    Conventions::RayDirection RayDirectionConvention = RayDirectionDefault>
    constexpr auto refract(const V& v, const N& n, T eta) {
        return refract<RayDirectionConvention, TIR>(v, n, eta);
    }
    template <Conventions::RayDirection RayDirectionConvention, bool TIR, ConceptVector V, ConceptVector N, typename T, typename U>
    constexpr auto refract(const V& v, const N& n, T ior_src, U ior_dest) {
        auto eta = ior_src/ior_dest;
        return Refract<V, N, decltype(eta), RayDirectionConvention, TIR> { v, n, eta };
    }
    template <
    ConceptVector V, ConceptVector N, typename T, typename U,
    bool TIR = true,
    Conventions::RayDirection RayDirectionConvention = RayDirectionDefault>
    constexpr auto refract(const V& v, const N& n, T ior_src, U ior_dest) {
        auto eta = ior_src/ior_dest;
        return refract<RayDirectionConvention, TIR>(v, n, eta);
    }

    template <ConceptVector A, ConceptVector B>
    constexpr auto scalar_projection(const A& a, const B& b) {
        return dot(a, normalize(b));
    }

    template <ConceptVector A, ConceptVector B>
    constexpr auto vector_projection(const A& a, const B& b) {
        return normalize(b) * scalar_projection(a, b);
    }

    template <ConceptVector A, ConceptVector B>
    constexpr auto vector_rejection(const A& a, const B& b) {
        return a - vector_projection(a, b);
    }

    template <ConceptVector A, ConceptVector B>
    constexpr auto orthonormalize(const A& a, const B& b) {
        return normalize(vector_rejection(a, b));
    }

    template <ConceptVector L, ConceptVector R, typename BinaryOperator, bool Indexed>
    struct VectorComponentWiseBinaryOperation {
        L left;
        R right;
        BinaryOperator op;

        constexpr VectorComponentWiseBinaryOperation(const L& l, const R& r, const BinaryOperator& op = {})
            : left(l), right(r), op(op)
        {
            assert_extent(left.size(), right.size(), std::equal_to<>{});
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType i) const {
            if constexpr(Indexed) {
                return op(left[i], right[i], i);
            } else {
                return op(left[i], right[i]);
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<VectorComponentWiseBinaryOperation<L, R, BinaryOperator, Indexed>>()[0])
        >;

        constexpr auto size() const { return left.size(); }
    };

    template <bool Indexed, ConceptVector L, ConceptVector R, typename BinaryOperator>
    constexpr auto binary_operation_component_wise(const L& l, const R& r, const BinaryOperator& op) {
        return VectorComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), decltype(op), Indexed> { l.ref(), r.ref(), op };
    }
    template <ConceptVector L, ConceptVector R, typename BinaryOperator, bool Indexed = false>
    constexpr auto binary_operation_component_wise(const L& l, const R& r, const BinaryOperator& op) {
        return binary_operation_component_wise<Indexed, L, R, BinaryOperator>(l, r, op);
    }

    template <ConceptVector L, ConceptVector R>
    constexpr auto operator* (const L& l, const R& r) {
        return VectorComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), std::multiplies<>, false>{ l.ref(), r.ref() };
    }
    template <ConceptVector L, ConceptVector R>
    constexpr auto operator/ (const L& l, const R& r) {
        return VectorComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), std::divides<>, false>{ l.ref(), r.ref() };
    }
    template <ConceptVector L, ConceptVector R>
    constexpr auto operator+ (const L& l, const R& r) {
        return VectorComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), std::plus<>, false>{ l.ref(), r.ref() }; 
    }
    template <ConceptVector L, ConceptVector R>
    constexpr auto operator- (const L& l, const R& r) {
        return VectorComponentWiseBinaryOperation<decltype(l.ref()), decltype(r.ref()), std::minus<>, false>{ l.ref(), r.ref() };
    }

    template <ConceptVector L, typename Field, typename BinaryOperator, bool Indexed>
    struct VectorScalarBinaryOperation {
        L left;
        Field right;
        BinaryOperator op;

        constexpr VectorScalarBinaryOperation(const L& l, const Field& r, const BinaryOperator& op = {})
            : left(l), right(r), op(op)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType i) const {
            if constexpr(Indexed) {
                return op(left[i], right, i);
            } else {
                return op(left[i], right);
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<VectorScalarBinaryOperation<L, Field, BinaryOperator, Indexed>>()[0])
        >;

        constexpr auto size() const { return left.size(); }
    };

    template <ConceptVector V, typename Field>
    constexpr auto operator* (const V& l, const Field& r) {
        return VectorScalarBinaryOperation<decltype(l.ref()), Field, std::multiplies<>, false>{ l.ref(), r };
    }
    template <typename Field, ConceptVector V>
    constexpr auto operator* (const Field& l, const V& r) { return r * l; }
    template <ConceptVector V, typename Field>
    requires (!ConceptVector<Field>)
    constexpr auto operator/ (const V& l, const Field& r) {
        return VectorScalarBinaryOperation<decltype(l.ref()), Field, std::divides<>, false>{ l.ref(), r };
    }
    template <typename Field, ConceptVector V> requires (!ConceptVector<Field> && !ConceptVectorStatic<V>)
    constexpr auto operator/ (const Field& l, const V& r) { return as_vector(l, r.size().get()) / r; }
    template <typename Field, ConceptVectorStatic V> requires (!ConceptVector<Field>)
    constexpr auto operator/ (const Field& l, const V& r) { return as_vector<size_static<V>()>(l) / r; }
    template <ConceptVector V, typename Field>
    constexpr auto operator+ (const V& l, const Field& r) {
        return VectorScalarBinaryOperation<decltype(l.ref()), Field, std::plus<>, false>{ l.ref(), r };
    }
    template <typename Field, ConceptVector V>
    constexpr auto operator+ (const Field& l, const V& r) { return r + l; }
    template <ConceptVector V, typename Field>
    constexpr auto operator- (const V& l, const Field& r) {
        return VectorScalarBinaryOperation<decltype(l.ref()), Field, std::minus<>, false>{ l.ref(), r };
    }
    template <ConceptVector L, typename R>
    constexpr auto& operator*= (L& l, const R& r) { l = l * r; return l; }
    template <ConceptVector L, typename R>
    constexpr auto& operator/= (L& l, const R& r) { l = l / r; return l; }
    template <ConceptVector L, typename R>
    constexpr auto& operator+= (L& l, const R& r) { l = l + r; return l; }
    template <ConceptVector L, typename R>
    constexpr auto& operator-= (L& l, const R& r) { l = l - r; return l; }

    template <bool Indexed, ConceptVector A, typename B, typename BinaryOperator>
    constexpr auto binary_operation(const A& v, const B& b, const BinaryOperator& op) {
        return VectorScalarBinaryOperation<decltype(v.ref()), B, decltype(op), Indexed> { v.ref(), b, op };
    }
    template <ConceptVector A, typename B, typename BinaryOperator, bool Indexed = false>
    constexpr auto binary_operation(const A& v, const B& b, const BinaryOperator& op) {
        return binary_operation<Indexed, A, B, BinaryOperator>(v, b, op);
    }

    template<ConceptVector A, typename B>
    constexpr auto step(const A& a, const B& b) {
        return binary_operation(a, b, [](auto x, auto y){ return x < y ? 0 : 1; });
    }

    template<ConceptVector A, typename B>
    constexpr auto min(const A& a, const B& b) {
        return binary_operation(a, b, [](auto x, auto y){ return x < y ? x : y; });
    }

    template<ConceptVector A, typename B>
    constexpr auto max(const A& a, const B& b) {
        return binary_operation(a, b, [](auto x, auto y){ return x < y ? y : x; });
    }

    template <ConceptVector A, typename B>
    constexpr auto euclidean_remainder(const A& a, const B& b) {
        return binary_operation(a, b, [](auto x, auto y){ return euclidean_remainder(x, y); });
    }

    template <ConceptVector A, ConceptVector B>
    constexpr auto euclidean_remainder(const A& a, const B& b) {
        return binary_operation_component_wise(a, b, [](auto x, auto y){ return euclidean_remainder(x, y); });
    }

    template <ConceptVector A, typename B, typename C, typename TernaryOperator, bool Indexed>
    struct VectorTernaryOperation {
        A vector;
        B b;
        C c;
        TernaryOperator op;

        constexpr VectorTernaryOperation(const A& a, const B& b, const C& c, const TernaryOperator& op = {})
            : vector(a), b(b), c(c), op(op)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            if constexpr(Indexed) {
                return op(vector[element], b, c, element);
            } else {
                return op(vector[element], b, c);
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<VectorTernaryOperation<A, B, C, TernaryOperator, Indexed>>()[0])
        >;

        constexpr auto size() const { return vector.size(); }
    };

    template <ConceptVector A, typename B, typename C>
    constexpr auto clamp(const A& v, const B& lower, const C& upper) {
        auto op = [](const auto& x, const auto& lower, const auto& upper)->auto {
            using std::clamp;
            using Maths::clamp;
            return clamp(x, lower, upper);
        };
        return VectorTernaryOperation<decltype(v.ref()), B, C, decltype(op), false> { v.ref(), lower, upper, {} };
    }

    template <ConceptVector A, ConceptVector B, ConceptVector C>
    constexpr auto clamp(const A& v, const B& lower, const C& upper) {
        auto op = [](const auto& x, const auto& lower, const auto& upper, auto element)->auto {
            using std::clamp;
            using Maths::clamp;
            return clamp(x, lower[element], upper[element]);
        };
        return VectorTernaryOperation<decltype(v.ref()), B, C, decltype(op), true> { v.ref(), lower, upper, {} };
    }

    template <bool Indexed, ConceptVector A, typename B, typename C, typename TernaryOperator>
    constexpr auto ternary_operation(const A& a, const B& b, const C& c, const TernaryOperator& op) {
        return VectorTernaryOperation<decltype(a.ref()), B, C, TernaryOperator, Indexed> { a.ref(), b, c, op };
    }
    template <ConceptVector A, typename B, typename C, typename TernaryOperator, bool Indexed = false>
    constexpr auto ternary_operation(const A& a, const B& b, const C& c, const TernaryOperator& op) {
        return ternary_operation<Indexed, A, B, C, TernaryOperator>(a, b, c, op);
    }

    template <ConceptVector V, ConceptMatrix BasisSource, ConceptMatrix BasisDestination>
    constexpr auto change_of_basis(const V& vector, const BasisSource& basis_src, const BasisDestination& basis_dest) {
        return mat_change_of_basis(basis_src, basis_dest) * vector;
    }

    template <ConceptVector V>
    constexpr auto min(const V& v) {
        using std::min;
        auto minimum = v[0];
        for(IndexType i = 1; i < v.size().get(); ++i)
            minimum = min(minimum, v[i]);
        return minimum;
    }

    template<ConceptVector A, ConceptVector B>
    constexpr auto min(const A& a, const B& b) {
        return binary_operation_component_wise(a, b, [](auto x, auto y){ using std::min; return min(x, y); });
    }

    template <ConceptVector V>
    constexpr auto max(const V& v) {
        using std::max;
        auto maximum = v[0];
        for(IndexType i = 1; i < v.size().get(); ++i)
            maximum = max(maximum, v[i]);
        return maximum;
    }

    template<ConceptVector A, ConceptVector B>
    constexpr auto max(const A& a, const B& b) {
        return binary_operation_component_wise(a, b, [](auto x, auto y){ using std::max; return max(x, y); });
    }

    template <ConceptVector V, typename Real>
    constexpr auto norm_p(const V& v, Real p) {
        using std::abs;
        auto sum = std::pow(abs(v[0]), p);
        for(IndexType i = 1; i < v.size().get(); ++i)
            sum += std::pow(abs(v[i]), p);
        return std::pow(sum, Real{1}/p);
    }

    template <ConceptVector V>
    constexpr auto norm_uniform(const V& v) {
        using std::abs;
        auto norm = abs(v[0]);
        for(IndexType i = 0; i < v.size().get(); ++i)
            if(norm < abs(v[i])) norm = abs(v[i]);
        return norm;
    }
    template <ConceptVector V>
    constexpr auto norm_maximum(const V& v) { return norm_uniform(v); }
    template <ConceptVector V>
    constexpr auto norm_max(const V& v) { return norm_uniform(v); }
    template <ConceptVector V>
    constexpr auto norm_chebyshev(const V& v) { return norm_uniform(v); }
    template <ConceptVector V>
    constexpr auto norm_infinity(const V& v) { return norm_uniform(v); }
    template <ConceptVector V>
    constexpr auto norm_inf(const V& v) { return norm_uniform(v); }

    template <ConceptVector V>
    constexpr auto norm_taxicab(const V& v) {
        using std::abs;
        auto sum = static_cast<typename V::value_type>(0);
        for(IndexType i = 0; i < v.size().get(); ++i)
                sum += abs(v[i]);
        return sum;
    }
    template <ConceptVector V>
    constexpr auto norm_manhattan(const V& v) { return norm_taxicab(v); }
    template <ConceptVector V>
    constexpr auto norm_p1(const V& v) { return norm_taxicab(v); }

    template <ConceptVector V>
    constexpr auto norm_frobenius(const V& v) {
        using std::sqrt;
        return sqrt(dot(v, v));
    }
    template <ConceptVector V>
    constexpr auto norm_euclidean(const V& v) { return norm_frobenius(v); }
    template <ConceptVector V>
    constexpr auto norm(const V& v) { return norm_frobenius(v); }
    template <ConceptVector V>
    constexpr auto magnitude(const V& v) { return norm_frobenius(v); }
    template <ConceptVector V>
    constexpr auto length(const V& v) { return norm_frobenius(v); }
    template <ConceptVector V>
    constexpr auto norm_p2(const V& v) { return norm_frobenius(v); }

    template <ConceptVector V>
    constexpr auto normalize_uniform(const V& v) { return v/norm_max(v); }
    template <ConceptVector V>
    constexpr auto normalize_maximum(const V& v) { return normalize_uniform(v); }
    template <ConceptVector V>
    constexpr auto normalize_max(const V& v) { return normalize_uniform(v); }
    template <ConceptVector V>
    constexpr auto normalize_chebyshev(const V& v) { return normalize_uniform(v); }
    template <ConceptVector V>
    constexpr auto normalize_infinity(const V& v) { return normalize_uniform(v); }
    template <ConceptVector V>
    constexpr auto normalize_inf(const V& v) { return normalize_uniform(v); }

    template <ConceptVector V>
    constexpr auto normalize_taxicab(const V& v) { return v/norm_taxicab(v); }
    template <ConceptVector V>
    constexpr auto normalize_manhattan(const V& v) { return normalize_taxicab(v); }
    template <ConceptVector V>
    constexpr auto normalize_p1(const V& v) { return normalize_taxicab(v); }

    template <ConceptVector V>
    constexpr auto normalize_frobenius(const V& v) { return v/norm_frobenius(v); }
    template <ConceptVector V>
    constexpr auto normalize_euclidean(const V& v) { return normalize_frobenius(v); }
    template <ConceptVector V>
    constexpr auto normalize_p2(const V& v) { return normalize_uniform(v); }
    template <ConceptVector V>
    constexpr auto normalize(const V& v) { return normalize_frobenius(v); }
    
    template <ConceptVector V, typename Real>
    constexpr auto normalize_p(const V& v, Real p) { return v/norm_p(v, p); }

    template <ConceptVector V>
    constexpr auto normalize_minmax(const V& v) {
        auto minimum = min(v);
        auto maximum = max(v);
        return (v - minimum)/(maximum - minimum);
    }

    template <ConceptVector A, ConceptVector B, typename T>
    constexpr auto spherical_linear_interpolation(const A& a, const B& b, const T& t) {
        using std::sin;
        using std::acos;
        auto omega = acos(inner_product_euclidean(a, b)/(length(a)*length(b)));
        auto sin_omega = sin(omega);
        bool reduce_to_lerp = !(sin_omega > std::numeric_limits<T>::epsilon());
        T t_a = reduce_to_lerp? T{1} - t : sin((T{1} - t)*omega);
        T t_b = reduce_to_lerp? t : sin(t*omega);
        T denom = reduce_to_lerp? T{1} : sin_omega;
        return (a*t_a + b*t_b)/denom;
    }
    template <ConceptVector A, ConceptVector B, typename T>
    constexpr auto slerp(const A& a, const B& b, const T& t) {
        return spherical_linear_interpolation(a, b, t);
    }

    template <ConceptVector V>
    struct Quaternion {
        V components;

        using tag_type = QuaternionTag;
        using value_type = V::value_type;

        template<ConceptVector U>
        constexpr Quaternion(const U& v) : components(v)
        {
            assert_extent(v.size(), StaticExtent<4>{}, std::equal_to<>{});
        }

        constexpr auto ref() const { return *this; }

        constexpr vec_static_t<3, value_type> vector() const {
            return {components[1], components[2], components[3]};
        }

        constexpr value_type scalar() const {
            return components[0];
        }

        constexpr auto operator[] (IndexType element) const {
            return components[element];
        }

        constexpr auto& operator[] (IndexType element) {
            return components[element];
        }
        
        template <ConceptVector U>
        constexpr auto& operator= (const U& v) {
            assert_extent(v.size(), this->size(), std::equal_to<>{});
            components = v;
            return *this;
        }

        constexpr auto size() const { return StaticExtent<4>{}; }
    };

    template <ConceptVector V>
    constexpr auto quat(const V& v) {
        return Quaternion<decltype(vec<4>(v.ref()))> { vec<4>(v.ref()) };
    }

    template <typename T>
    constexpr auto quat(const T& a, const T& b, const T& c, const T& d) {
        return quat(vec_static_t<4, T>{a, b, c, d});
    }

    template <ConceptVector V>
    constexpr auto quat_ref(const V& v) {
        return Quaternion<decltype(v.ref())> { v.ref() };
    }

    template <typename T>
    using quat_t = Quaternion<vec_static_t<4, T>>;

    template <ConceptVector V, typename T>
    constexpr auto quat_axis_angle(const V& axis, T angle) {
        using std::sin;
        using std::cos;
        T half_angle = angle * T{0.5};
        auto sine = sin(half_angle);
        return quat(vec_ref({cos(half_angle), axis[0] * sine, axis[1] * sine, axis[2] * sine}));
    }
    
    template <ConceptQuaternion Q>
    constexpr auto conjugate(const Q& q) {
        return quat_ref(unary_operation<true>(q, [](const auto& x, const auto& i){ return i==0? x : -x; }));
    }
    template <ConceptQuaternion Q>
    constexpr auto conj(const Q& q) { return conjugate(q); }
    
    template <ConceptQuaternion Q>
    constexpr auto inverse(const Q& q) {
        return quat_ref(conjugate(q) / inner_product(q, q));
    }

    template <ConceptQuaternion Q, ConceptVector V>
    struct QuaternionSandwichProduct {
        Q quaternion;
        V vector;

        decltype(inner_product(quaternion, quaternion)) scale;

        using tag_type = QuaternionTag;

        constexpr QuaternionSandwichProduct(const Q& q, const V& v) : quaternion(q), vector(v)
        {
            assert_extent(v.size(), StaticExtent<3>{}, std::equal_to<>{});

            scale = inner_product(quaternion, quaternion);
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            constexpr typename V::value_type two {2};
            const auto &a = quaternion[0], &b = quaternion[1], &c = quaternion[2], &d = quaternion[3];
            const auto &x = vector[0], &y = vector[1], &z = vector[2];
            switch(element) {
                case 0: return x*scale + two*(-x*(c*c + d*d) + y*(b*c - d*a) + z*(b*d + c*a));
                case 1: return y*scale + two*(x*(b*c + d*a) - y*(b*b + d*d) + z*(c*d - b*a));
                default: return z*scale + two*(x*(b*d - c*a) + y*(c*d + b*a) - z*(b*b + c*c));
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<QuaternionSandwichProduct<Q,V>>()[0])
        >;

        constexpr auto size() const { return StaticExtent<3>{}; }
    };
    
    template <ConceptQuaternion L, ConceptVector R>
    constexpr auto operator* (const L& q, const R& v) {
        assert_extent(v.size(), StaticExtent<3>{}, std::equal_to<>{});
        return QuaternionSandwichProduct<L, R> { q, v };
    }
    template <ConceptVector L, ConceptQuaternion R>
    constexpr auto operator* (const L& v, const R& q) { return inverse(q) * v; }

    template <ConceptQuaternion L, ConceptQuaternion R>
    struct HamiltonProduct {
        L left;
        R right;

        using tag_type = QuaternionTag;

        constexpr HamiltonProduct(const L& r, const R& s) : left(r), right(s) {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            const auto &a1 = left[0], &b1 = left[1], &c1 = left[2], &d1 = left[3];
            const auto &a2 = right[0], &b2 = right[1], &c2 = right[2], &d2 = right[3];
            switch(element) {
                case 0: return a1*a2 - b1*b2 - c1*c2 - d1*d2;
                case 1: return a1*b2 + b1*a2 + c1*d2 - d1*c2;
                case 2: return a1*c2 - b1*d2 + c1*a2 + d1*b2;
                default: return a1*d2 + b1*c2 - c1*b2 + d1*a2;
            }
        }

        using value_type = invoke_expression_template_result_t<
            decltype(std::declval<HamiltonProduct<L,R>>()[0])
        >;

        constexpr auto size() const { return StaticExtent<4>{}; }
    };
    
    template <ConceptQuaternion L, ConceptQuaternion R>
    constexpr auto operator* (const L& r, const R& s) {
        return HamiltonProduct<L,R> { r,s };
    }
    template <ConceptQuaternion L, ConceptQuaternion R>
    constexpr auto& operator*= (L& l, const R& r) { l = r * l; return l; }

    template <ConceptQuaternion Q, ConceptExtent ExtR, ConceptExtent ExtC, bool ColumnMajor>
    struct QuaternionAsMatrix {
        Q quaternion;
        ExtR rows;
        ExtC columns;

        decltype(inner_product(quaternion, quaternion)) scale;

        constexpr static bool column_major = ColumnMajor;

        using value_type = typename Q::value_type;

        constexpr QuaternionAsMatrix(const Q& q, const ExtR& r = {}, const ExtR& c = {})
            : quaternion(q), rows(r), columns(c)
        {
            assert_extent(r, StaticExtent<3>{}, std::greater_equal<>{});
            assert_extent(c, StaticExtent<3>{}, std::greater_equal<>{});

            scale = inner_product(quaternion, quaternion);
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType row, IndexType column) const {
            constexpr value_type two {2};
            const auto
                &a = quaternion[0],
                &b = quaternion[1],
                &c = quaternion[2],
                &d = quaternion[3];
            switch(row) {
                case 0: {
                    switch(column) {
                        case 0: return scale - two*(c*c + d*d);
                        case 1: return two*(b*c - d*a);
                        case 2: return two*(b*d + c*a);
                    }
                }
                case 1: {
                    switch(column) {
                        case 0: return two*(b*c + d*a);
                        case 1: return scale - two*(b*b + d*d);
                        case 2: return two*(c*d - b*a);
                    }
                }
                case 2: {
                    switch(column) {
                        case 0: return two*(b*d - c*a);
                        case 1: return two*(c*d + b*a);
                        case 2: return scale - two*(b*b + c*c);
                    }
                }
            }
            return kronecker_delta<value_type>(row, column);
        }

        constexpr auto row_count() const { return rows; }
        constexpr auto column_count() const { return columns; }
    };
    
    template <IndexType Rows, IndexType Columns, bool ColumnMajor, ConceptQuaternion Q>
    constexpr auto as_matrix(const Q& q) {
        return QuaternionAsMatrix<Q, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor> { q };
    }
    template <IndexType Rows, IndexType Columns, ConceptQuaternion Q, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_matrix(const Q& q) { return as_matrix<Rows, Columns, ColumnMajor, Q>(q); }
    template <ConceptQuaternion Q, IndexType Rows = 3, IndexType Columns = 3, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_matrix(const Q& q) { return as_matrix<Rows, Columns, ColumnMajor, Q>(q); }

    template <bool ColumnMajor, ConceptQuaternion Q>
    constexpr auto as_matrix(const Q& q, IndexType rows, IndexType columns) {
        return QuaternionAsMatrix<Q, DynamicExtent, DynamicExtent, ColumnMajor> { q, rows, columns };
    }
    template <ConceptQuaternion Q, bool ColumnMajor = ColumnMajorDefault>
    constexpr auto as_matrix(const Q& q, IndexType rows, IndexType columns) { return as_matrix<ColumnMajor, Q>(q, rows, columns); }

    template <ConceptMatrix M>
    struct MatrixAsQuaternion {
        M matrix;
        std::remove_const_t<typename M::value_type> scale;

        using tag_type = QuaternionTag;
        using value_type = typename M::value_type;

        constexpr MatrixAsQuaternion(const M& m) : matrix(m)
        {
            using std::pow;
            assert_extent(m.row_count(), StaticExtent<3>{}, std::equal_to<>{});
            assert_extent(m.column_count(), StaticExtent<3>{}, std::equal_to<>{});

            scale = pow(determinant(matrix), decltype(scale){1/3.0});
        }

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            using std::sqrt;
            using std::max;
            using std::copysign;
            constexpr value_type zero {0};
            constexpr value_type two {2};
            switch(element) {
                case 0: {
                    return sqrt(max(zero, scale + matrix[0,0] + matrix[1,1] + matrix[2,2])) / two;
                }
                case 1: {
                    return copysign(sqrt(max(zero, scale + matrix[0,0] - matrix[1,1] - matrix[2,2])) / two, matrix[2,1]-matrix[1,2]);
                }
                case 2: {
                    return copysign(sqrt(max(zero, scale - matrix[0,0] + matrix[1,1] - matrix[2,2])) / two, matrix[0,2]-matrix[2,0]);
                }
                default: {
                    return copysign(sqrt(max(zero, scale - matrix[0,0] - matrix[1,1] + matrix[2,2])) / two, matrix[1,0]-matrix[0,1]);
                }
            }
        }

        constexpr auto size() const { return StaticExtent<4>{}; }
    };

    template <ConceptMatrix M>
    constexpr auto as_quaternion(const M& m) {
        return MatrixAsQuaternion<decltype(m.ref())> { m.ref() };
    }
    template <ConceptMatrix M>
    constexpr auto as_quat(const M& m) { return as_quaternion(m); }

    template <ConceptQuaternion Q, ConceptMatrix BasisMap, typename T>
    constexpr auto change_of_basis(const Q& quaternion, const BasisMap& transition_matrix, const T& transition_matrix_determinant) {
        return quat(join(quaternion.scalar(), transition_matrix * quaternion.vector() * transition_matrix_determinant));
    }

    template <ConceptQuaternion Q, ConceptMatrix BasisSource, ConceptMatrix BasisDestination>
    constexpr auto change_of_basis(const Q& quaternion, const BasisSource& basis_src, const BasisDestination& basis_dest) {
        auto M = mat_change_of_basis(basis_src, basis_dest);
        return change_of_basis(quaternion, M, determinant(M));
    }

    template <ConceptQuaternion A, ConceptQuaternion B>
    constexpr auto slerp_flip(const A& src_quat, const B& ref_quat) {
        if(inner_product_euclidean(src_quat, ref_quat) < 0)
            return quat(-src_quat);
        return quat(src_quat);
    }

    template <ConceptVector F, ConceptVector U>
    constexpr auto quat_look_rotation(const F& forward, const U& up) {
        auto up_orthonormal = orthonormalize(up, forward);
        auto right = cross(up_orthonormal, forward);
        return as_quaternion(join(join(as_column(right), as_column(up_orthonormal)), as_column(forward)));
    }

    template <ConceptVector V, bool InverseTransform>
    struct HypersphericalCoordinates {
        V coordinates;

        using value_type = V::value_type;

        constexpr HypersphericalCoordinates(const V& v)
            : coordinates(v)
        {}

        constexpr auto ref() const { return *this; }

        constexpr auto operator[] (IndexType element) const {
            if constexpr(InverseTransform) {
                using std::sin;
                using std::cos;
                auto r = coordinates[0];
                const auto& phi = coordinates;
                if constexpr(ConceptVectorStatic<V> && size_static<V>() <= 3) {
                    if constexpr(size_static<V>() == 1) {
                        return r;
                    } else if constexpr(size_static<V>() == 2) {
                        switch (element) {
                            case 0: return r*cos(phi[1]);
                            case 1: return r*sin(phi[1]);
                            default: return value_type{0};
                        }
                    } else if constexpr(size_static<V>() == 3) {
                        switch (element) {
                            case 0: return r*cos(phi[1]);
                            case 1: return r*sin(phi[1])*cos(phi[2]);
                            case 2: return r*sin(phi[1])*sin(phi[2]);
                            default: return value_type{0};
                        }
                    }
                } else {
                    auto x_n = r;
                    auto n = element+1;
                    if(n < size().get()) {
                        for(IndexType i = 1; i < n; ++i)
                            x_n *= sin(phi[i]);
                        x_n *= cos(phi[n]);
                    } else {
                        for(IndexType i = 1; i < size().get(); ++i)
                            x_n *= sin(phi[i]);
                    }
                    return x_n;
                }
            } else {
                using std::sqrt;
                using std::atan2;
                if constexpr(ConceptVectorStatic<V> && size_static<V>() <= 3) {
                    if constexpr(size_static<V>() == 1) {
                        return coordinates[0];
                    } else if constexpr(size_static<V>() == 2) {
                        switch (element) {
                            case 0: return norm_frobenius(coordinates);
                            case 1: return atan2(coordinates[1], coordinates[0]);
                            default: return value_type{0};
                        }
                    } else if constexpr(size_static<V>() == 3) {
                        using std::acos;
                        switch (element) {
                            case 0: return norm_frobenius(coordinates);
                            case 1: return acos(coordinates[0]/norm_frobenius(coordinates));
                            case 2: return atan2(coordinates[2], coordinates[1]);
                            default: return value_type{0};
                        }
                    }
                } else {
                    if(element == 0)
                        return norm_frobenius(coordinates);
                    auto sum = value_type{0};
                    for (IndexType i = element; i < size().get(); ++i)
                        sum += coordinates[i]*coordinates[i];
                    return atan2(sqrt(sum), coordinates[element-1]);
                }
            }
        }

        constexpr auto size() const { return coordinates.size(); }
    };

    template <ConceptVector V>
    constexpr auto hyperspherical_to_cartesian(const V& v) {
        return HypersphericalCoordinates<decltype(v.ref()), true> { v.ref() };
    }
    template <ConceptVector V>
    constexpr auto cartesian_to_hyperspherical(const V& v) {
        return HypersphericalCoordinates<decltype(v.ref()), false> { v.ref() };
    }

    template <ConceptMatrix M>
    inline void print(const M& mat, std::ostream& os = std::cout, std::streamsize spacing_width = 12) {
        for (IndexType row = 0; row < mat.row_count().get(); ++row) {
            for (IndexType col = 0; col < mat.column_count().get(); ++col)
                os << std::setw(spacing_width) << mat[row, col] << ",";
            os << std::endl;
        }
    }

    template <ConceptVector V>
    inline void print(const V& vec, std::ostream& os = std::cout) {
        if(!vec.size().get()) {
            os << "()" << std::endl;
            return;
        }
        std::cout << "(";
        for (IndexType i = 0; i < vec.size().get()-1; ++i)
            os << vec[i] << ", ";
        os << vec[vec.size().get()-1];
        os << ")" << std::endl;
    }

} // namespace Maths
