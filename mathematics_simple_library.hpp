#pragma once

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

namespace Maths {

	using IndexType = std::size_t;

	struct none_t {};

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

	//NOTE: may return negative zero
	template <typename T>
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
	template <typename T>
	constexpr T euclidean_modulo(T a, T b) { return euclidean_remainder(a, b); }
	template <typename T>
	constexpr T eucmod(T a, T b) { return euclidean_remainder(a, b); }

	template <typename T, typename U>
	constexpr T linear_interpolation(T a, T b, U t) {
		return a*(static_cast<U>(1) - t) + b*t;
	}
	template <typename T, typename U>
	constexpr T lerp(T a, T b, U t) { return linear_interpolation(a, b, t); }

	template <typename T>
	constexpr T kronecker_delta(IndexType i, IndexType j) { return static_cast<T>(i==j); }

	template <class T>
	concept Container = requires(T& container) {
		typename T::value_type;
		{std::data(container)};
		{std::size(container)};
	};

	template <typename E>
	concept Extent = requires (const E& extent) {
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

	template <Extent E, Extent F, typename ComparisonOperator>
    requires (E::is_static() && F::is_static())
	constexpr void assert_extent(const E&, const F&, ComparisonOperator cmp) {
		static_assert(cmp(E::get(), F::get()));
	}

	template <Extent E, Extent F, typename ComparisonOperator>
	void assert_extent([[maybe_unused]] const E& extent, [[maybe_unused]] const F& other, [[maybe_unused]] ComparisonOperator cmp) {
		assert(cmp(extent.get(), other.get()));
	}

	template <Extent E, Extent F, typename ComparisonOperator>
	constexpr auto compare_extent([[maybe_unused]] const E& extent, [[maybe_unused]] const F& other, [[maybe_unused]] ComparisonOperator cmp) {
		return cmp(extent.get(), other.get());
	}

	template <IndexType Value, Extent E, typename BinaryOperator>
	constexpr auto evaluate_extent([[maybe_unused]] const E& extent, [[maybe_unused]] BinaryOperator op) {
		if constexpr(E::is_static())
			return op(extent, StaticExtent<Value>{});
		else
			return op(extent, Value);
	}

	template <Extent E, Extent F, typename BinaryOperator>
	constexpr auto evaluate_extent([[maybe_unused]] const E& extent, [[maybe_unused]] const F& other, [[maybe_unused]] BinaryOperator op) {
		if constexpr((E::is_static() && F::is_static()) || !(E::is_static() || F::is_static()))
			return op(extent, other);
		else if constexpr(E::is_static() && !F::is_static())
			return op(DynamicExtent{E::get()}, other);
		else if constexpr(!E::is_static() && F::is_static())
			return op(extent, DynamicExtent{F::get()});
	}
	
	template <typename V>
	concept Vector = requires(const V& vector, IndexType element) {
		{ vector[element] };
		{ vector.size() } -> Extent;
	};
	
	template <typename V>
	concept VectorStatic = Vector<V> && requires(const V& vector) {
		{decltype(vector.size())::get()};
	};

	template <typename M>
	concept Matrix = requires(const M& matrix, IndexType row, IndexType column) {
		{ matrix[row, column] };
		{ matrix.row_count() } -> Extent;
		{ matrix.column_count() } -> Extent;
	};

	template <typename M>
	concept MatrixStatic = Matrix<M> && requires(const M& matrix) {
		{decltype(matrix.row_count())::get()};
		{decltype(matrix.column_count())::get()};
	};

	template <Matrix M>
	constexpr auto row_count_static() { return decltype(std::declval<M>().row_count())::get(); }
	template <Matrix M>
	constexpr auto column_count_static() { return decltype(std::declval<M>().column_count())::get(); }
	
	enum class MatrixIdentityType {
		Additive,
		Multiplicative,
		Hadamard
	};

	template <typename Field, Extent ExtR, Extent ExtC, MatrixIdentityType Type>
	struct MatrixIdentity {
		ExtR rows;
		ExtC columns;

		constexpr MatrixIdentity(const ExtR& rows = {}, const ExtC& columns = {})
			: rows(rows), columns(columns)
		{}

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

	template <IndexType Rows, IndexType Columns, typename T = float>
	constexpr auto mat_additive_identity() {
		return MatrixIdentity<T, StaticExtent<Rows>, StaticExtent<Columns>, MatrixIdentityType::Additive> { {}, {} };
	}
	template <typename T = float>
	constexpr auto mat_additive_identity(IndexType rows, IndexType columns) {
		return MatrixIdentity<T, DynamicExtent, DynamicExtent, MatrixIdentityType::Additive> { rows, columns };
	}
	template <IndexType Rows, IndexType Columns, typename T = float>
	constexpr auto mat_zero() {
		return mat_additive_identity<Rows, Columns, T>();
	}
	template <typename T = float>
	constexpr auto mat_zero(IndexType rows, IndexType columns) {
		return mat_additive_identity<T>(rows, columns);
	}

	template <IndexType Rows, IndexType Columns, typename T = float>
	constexpr auto mat_multiplicative_identity() {
		return MatrixIdentity<T, StaticExtent<Rows>, StaticExtent<Columns>, MatrixIdentityType::Multiplicative> { {}, {} };
	}
	template <typename T = float>
	constexpr auto mat_multiplicative_identity(IndexType rows, IndexType columns) {
		return MatrixIdentity<T, DynamicExtent, DynamicExtent, MatrixIdentityType::Multiplicative> { rows, columns };
	}
	template <IndexType Rows, IndexType Columns, typename T = float>
	constexpr auto mat_identity() {
		return mat_multiplicative_identity<Rows, Columns, T>();
	}
	template <typename T = float>
	constexpr auto mat_identity(IndexType rows, IndexType columns) {
		return mat_multiplicative_identity<T>(rows, columns);
	}

	template <IndexType Rows, IndexType Columns, typename T = float>
	constexpr auto mat_hadamard_identity() {
		return MatrixIdentity<T, StaticExtent<Rows>, StaticExtent<Columns>, MatrixIdentityType::Hadamard> { {}, {} };
	}
	template <typename T = float>
	constexpr auto mat_hadamard_identity(IndexType rows, IndexType columns) {
		return MatrixIdentity<T, DynamicExtent, DynamicExtent, MatrixIdentityType::Hadamard> { rows, columns };
	}
	template <IndexType Rows, IndexType Columns, typename T = float>
	constexpr auto mat_one() {
		return mat_hadamard_identity<Rows, Columns, T>();
	}
	template <typename T = float>
	constexpr auto mat_one(IndexType rows, IndexType columns) {
		return mat_hadamard_identity<T>(rows, columns);
	}

	template <typename Field, Extent ExtR, Extent ExtC, bool ColumnMajor>
	struct MatrixRef {
		Field* base_ptr;
		ExtR rows;
		ExtC columns;

		constexpr MatrixRef(Field* base_ptr, const ExtR& rows = {}, const ExtC& columns = {})
			: base_ptr(base_ptr), rows(rows), columns(columns)
		{}

		template <Matrix M>
		constexpr void set_from(const M& m) const {
			auto
				rows_copy_extent = std::min(row_count().get(), m.row_count().get()),
				column_copy_extent = std::min(column_count().get(), m.column_count().get());
			for (IndexType row = 0; row < rows_copy_extent; ++row)
				for (IndexType column = 0; column < column_copy_extent; ++column)
					(*this)[row, column] = m[row, column];
		}

		template <Matrix M>
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

	template <typename T, bool ColumnMajor = false>
	constexpr auto mat_ref(T* base_ptr, IndexType rows, IndexType columns) {
		return MatrixRef<T, DynamicExtent, DynamicExtent, ColumnMajor> { base_ptr, rows, columns };
	}

	template <IndexType Rows, IndexType Columns, typename T = float, bool ColumnMajor = false>
	constexpr auto mat_ref(T* base_ptr) {
		return MatrixRef<T, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor> { base_ptr, {}, {} };
	}

	template <typename T, bool ColumnMajor = false>
	constexpr auto mat_ref(const T* base_ptr, IndexType rows, IndexType columns) {
		return MatrixRef<const T, DynamicExtent, DynamicExtent, ColumnMajor> { base_ptr, rows, columns };
	}

	template <IndexType Rows, IndexType Columns, typename T = float, bool ColumnMajor = false>
	constexpr auto mat_ref(const T* base_ptr) {
		return MatrixRef<const T, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor> { base_ptr, {}, {} };
	}

	template <IndexType Rows, IndexType Columns, typename T = float, bool ColumnMajor = false>
	constexpr auto mat_ref(std::initializer_list<T> elements) {
		return mat_ref<Rows, Columns, const T, ColumnMajor>(std::data(elements));
	}

	template <IndexType Rows, IndexType Columns, Container T, bool ColumnMajor = false>
	constexpr auto mat_ref(T& container) {
		return mat_ref<Rows, Columns, typename T::value_type, ColumnMajor>(std::data(container));
	}

	template <Container T, bool ColumnMajor = false>
	constexpr auto mat_ref(T& container, IndexType rows, IndexType columns) {
		return mat_ref<typename T::value_type, ColumnMajor>(std::data(container), rows, columns);
	}

	template <IndexType Rows, IndexType Columns, Container T, bool ColumnMajor = false>
	using mat_ref_static_container_t = decltype(mat_ref<Rows, Columns, T, ColumnMajor>(*static_cast<T*>(0)));

	template <IndexType Rows, IndexType Columns, typename T, bool ColumnMajor = false>
	using mat_ref_static_t = decltype(mat_ref<Rows, Columns, T, ColumnMajor>(std::initializer_list<T>{}));

	template <Container T, bool ColumnMajor = false>
	using mat_ref_dynamic_container_t = decltype(mat_ref<T, ColumnMajor>(*static_cast<T*>(0), 0, 0));

	template <typename T, bool ColumnMajor = false>
	using mat_ref_dynamic_t = decltype(mat_ref<T, ColumnMajor>(std::initializer_list<T>{}, 0, 0));

    template <typename T, IndexType Rows, IndexType Columns, bool ColumnMajor>
    struct MatrixObjectStatic {
        std::array<T, Rows * Columns> data;

        MatrixObjectStatic() = default;
        MatrixObjectStatic(const MatrixObjectStatic&) = default;
        MatrixObjectStatic(std::initializer_list<T> elements) {
            std::copy(elements.begin(), elements.end(), data.begin());
        }

        template <Matrix M>
        MatrixObjectStatic(const M& m) { *this = m; }

        template <typename U>
        using RefType = MatrixRef<U, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor>;

        auto ref() const { return RefType<const T> { std::data(data) }; }
        auto ref() { return RefType<T> { std::data(data) }; }

        template <Matrix M>
        MatrixObjectStatic& operator= (const M& m) { ref() = m; return *this; }

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

        MatrixObjectDynamic()
            : MatrixObjectDynamic(0, 0)
        {}

        MatrixObjectDynamic(const MatrixObjectDynamic&) = default;
        MatrixObjectDynamic(MatrixObjectDynamic&& other) { *this = std::move(other); }

        template <Matrix M>
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

		template <Matrix M>
        MatrixObjectDynamic& operator= (const M& m) {
            resize(m.row_count().get(), m.column_count().get());
            ref() = m;
            return *this;
        }

        template <typename U>
        using RefType = MatrixRef<U, DynamicExtent, DynamicExtent, ColumnMajor>;

        auto ref() const { return RefType<const T> { std::data(data), rows, columns }; }
        auto ref() { return RefType<T> { std::data(data), rows, columns }; }

        auto& operator[] (IndexType row, IndexType column) { return ref()[row, column]; }
        const auto& operator[] (IndexType row, IndexType column) const { return ref()[row, column]; }
        auto row_count() const { return ref().row_count(); }
        auto column_count() const { return ref().column_count(); }
    };

	template <IndexType Rows, IndexType Columns, typename T = float, bool ColumnMajor = false>
	inline auto mat(std::initializer_list<T> elements) {
		return MatrixObjectStatic<T, Rows, Columns, ColumnMajor> { elements };
	}

	template <IndexType Rows, IndexType Columns, Container T, bool ColumnMajor = false>
	inline auto mat(const T& container) {
		return MatrixObjectStatic<T, Rows, Columns, ColumnMajor> { { container } };
	}

	template <typename T, bool ColumnMajor = false>
	inline auto mat(std::initializer_list<T> elements, IndexType rows, IndexType columns) {
		return MatrixObjectDynamic<T, ColumnMajor> { elements, rows, columns };
	}

	template <Container T, bool ColumnMajor = false>
	inline auto mat(const T& container, IndexType rows, IndexType columns) {
		return MatrixObjectDynamic<T, ColumnMajor> { container, rows, columns };
	}

	template <IndexType Rows, IndexType Columns, typename T, bool ColumnMajor = false>
	using mat_static_t = MatrixObjectStatic<T, Rows, Columns, ColumnMajor>;

	template <typename T, bool ColumnMajor = false>
	using mat_dynamic_t = MatrixObjectDynamic<T, ColumnMajor>;

	template <Matrix L, Matrix R>
	struct AugmentedMatrix {
		L left;
		R right;

		constexpr AugmentedMatrix(const L& l, const R& r) : left(l), right(r) {
			assert_extent(left.row_count(), right.row_count(), std::equal_to<>{});
		}

		constexpr auto operator[] (IndexType row, IndexType column) const {
			if(column >= left.column_count().get())
				return right[row, column - left.column_count().get()];
			return left[row, column];
		}

		constexpr auto row_count() const { return left.row_count(); }
		constexpr auto column_count() const { return evaluate_extent(left.column_count(), right.column_count(), std::plus<>{}); }
	};

	template <Matrix L, Matrix R> constexpr auto augment(const L& left, const R& right) { return AugmentedMatrix{left, right}; }
	template <Matrix L, Matrix R> constexpr auto join(const L& left, const R& right) { return augment(left, right); }

	template <Matrix M, Extent E, bool Vertical = false, bool SecondHalf = false>
	struct SplitMatrix {
		M matrix;
		E split_bound;

		constexpr SplitMatrix(const M& m, const E& split_bound = {}) : matrix(m), split_bound(split_bound) {
			if constexpr (Vertical) {
				assert_extent(split_bound, matrix.row_count(), std::less<>{});
			} else {
				assert_extent(split_bound, matrix.column_count(), std::less<>{});
			}
		}

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

	template <Matrix M> constexpr auto split_left(const M& m, IndexType split_bound) {
		return SplitMatrix<M, DynamicExtent, false, false>{m, split_bound};
	}
	template <IndexType SplitBound, Matrix M> constexpr auto split_left(const M& m) {
		return SplitMatrix<M, StaticExtent<SplitBound>, false, false>{m};
	}

	template <Matrix M> constexpr auto split_right(const M& m, IndexType split_bound) {
		return SplitMatrix<M, DynamicExtent, false, true>{m, split_bound};
	}
	template <IndexType SplitBound, Matrix M> constexpr auto split_right(const M& m) {
		return SplitMatrix<M, StaticExtent<SplitBound>, false, true>{m};
	}

	template <Matrix M> constexpr auto split_top(const M& m, IndexType split_bound) {
		return SplitMatrix<M, DynamicExtent, true, false>{m, split_bound};
	}
	template <IndexType SplitBound, Matrix M> constexpr auto split_top(const M& m) {
		return SplitMatrix<M, StaticExtent<SplitBound>, true, false>{m};
	}

	template <Matrix M> constexpr auto split_bottom(const M& m, IndexType split_bound) {
		return SplitMatrix<M, DynamicExtent, true, true>{m, split_bound};
	}
	template <IndexType SplitBound, Matrix M> constexpr auto split_bottom(const M& m) {
		return SplitMatrix<M, StaticExtent<SplitBound>, true, true>{m};
	}
	
	template <Matrix M>
	struct ReducedRowEchelonMatrix {
		using value_type = std::remove_reference_t<decltype(std::declval<M>()[0,0])>;
		mat_dynamic_t<value_type> matrix;

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

		auto operator[] (IndexType row, IndexType column) const {
			return matrix[row, column];
		}
		auto row_count() const { return matrix.row_count(); }
		auto column_count() const { return matrix.column_count(); }
	};

	template <Matrix M> inline auto reduced_row_echelon_form(const M& m) { return ReducedRowEchelonMatrix{m}; }
	template <Matrix M> inline auto rref(const M& m) { return reduced_row_echelon_form(m); }

	template <Matrix M>
	struct Conjugate {
		M matrix;

		constexpr auto operator[] (IndexType row, IndexType column) const {
			using value_type = std::remove_reference_t<decltype(matrix[0,0])>;
			using std::conj;
			if constexpr(is_complex_v<value_type>)
				return conj(matrix[row, column]);
			else
				return matrix[row, column];
		}

		constexpr auto row_count() const { return matrix.row_count(); }
		constexpr auto column_count() const { return matrix.column_count(); }
	};

	template <Matrix M> constexpr auto conjugate(const M& m) { return Conjugate<M> {m}; }
	template <Matrix M> constexpr auto conj(const M& m) { return conjugate(m); }
	
	template <Matrix M>
	struct Transpose {
		M matrix;
		constexpr auto operator[] (IndexType row, IndexType column) const { return matrix[column, row]; }
		constexpr auto row_count() const { return matrix.column_count(); }
		constexpr auto column_count() const { return matrix.row_count(); }
	};

	template <Matrix M> constexpr auto transpose(const M& m) { return Transpose<M> { m }; }

	template <Matrix M> constexpr auto transpose_hermitian(const M& m) { return transpose(conjugate(m)); }

	template <Matrix M> constexpr auto gramian(const M& m) { return transpose_hermitian(m)*m; }
	template <Matrix M> constexpr auto gram(const M& m) { return gramian(m); }

	template <MatrixStatic M, Extent ErasedRow, Extent ErasedColumn>
	struct Submatrix {
		M matrix;
		ErasedRow erased_row;
		ErasedColumn erased_column;

		constexpr Submatrix(const M& matrix, const ErasedRow& erased_row = {}, const ErasedColumn& erased_column = {})
			: matrix(matrix), erased_row(erased_row), erased_column(erased_column)
		{
			assert_extent(erased_row, matrix.row_count(), std::less<>{});
			assert_extent(erased_column, matrix.column_count(), std::less<>{});
		}

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

	template <Matrix M>
	struct SubmatrixDynamic {
		M matrix;
		std::vector<IndexType> erased_rows;
		std::vector<IndexType> erased_columns;

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

	template <IndexType ErasedRow, IndexType ErasedColumn, MatrixStatic M>
	constexpr auto submatrix(const M& m) {
		return Submatrix<M, StaticExtent<ErasedRow>, StaticExtent<ErasedColumn>> { m };
	}

	template <Matrix M>
	constexpr auto submatrix(const M& m, IndexType erased_row, IndexType erased_column) {
		return SubmatrixDynamic<M> { m, erased_row, erased_column };
	}

	template <Matrix M>
	constexpr auto submatrix(const SubmatrixDynamic<M>& m, IndexType erased_row, IndexType erased_column) {
		return SubmatrixDynamic<M> { m, erased_row, erased_column };
	}

	template <Extent Rows, Extent Columns, Matrix M>
	constexpr auto determinant(const M& m) {
		assert_extent(m.row_count(), m.column_count(), std::equal_to<>{});
		using value_type = std::remove_reference_t<decltype(m[0,0])>;
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

	template <MatrixStatic M>
	constexpr auto determinant(const M& m) {
		return determinant<decltype(std::declval<M>().row_count()), decltype(std::declval<M>().column_count())>(m);
	}

	template <Matrix M>
	constexpr auto determinant(const M& m) {
		return determinant<DynamicExtent, DynamicExtent>(m);
	}
	template <Matrix M>
	constexpr auto det(const M& m) { return determinant(m); }

	template <Matrix M>
	struct Cofactor {
		M matrix;

		constexpr auto operator[] (IndexType row, IndexType column) const {
			using value_type = std::remove_reference_t<decltype(matrix[0,0])>;
			return static_cast<value_type>((row+column)&1?-1:1) * determinant(submatrix(matrix, row, column));
		}
		constexpr auto row_count() const { return matrix.row_count(); }
		constexpr auto column_count() const { return matrix.column_count(); }
	};

	template <Matrix M>
	constexpr auto cofactor(const M& m) { return Cofactor<M> { m }; }
	
	template <Matrix M>
	constexpr auto adjugate(const M& m) { return transpose(cofactor(m)); }
	
	template <Matrix M>
	constexpr auto inverse(const M& m) { return adjugate(m)/determinant(m); }
	template <Matrix M>
	constexpr auto inv(const M& m) { return inverse(m); }

	template <Matrix M>
	constexpr auto inverse_gauss_jordan(const M& m) {
        auto r = rref(augment(m, mat_identity(m.row_count().get(), m.column_count().get())));
		return split_right(r, m.column_count().get());
	}
	
	template <Matrix M, Extent E>
	struct RowOf {
		M matrix;
		E row;

		constexpr RowOf(const M& matrix, const E& row = {}) : matrix(matrix), row(row) {
			assert_extent(row, matrix.row_count(), std::less<>{});
		}

		constexpr auto operator[] (IndexType element) const { return matrix[row.get(), element]; }

		constexpr auto size() const { return matrix.column_count(); }
	};

	template <Matrix M>
	constexpr auto row_of(const M& m, IndexType row) { return RowOf<M, DynamicExtent> { m, row }; }
	
	template <IndexType Row, Matrix M>
	constexpr auto row_of(const M& m) { return RowOf<M, StaticExtent<Row>> { m }; }

	template <Matrix M>
	constexpr auto column_of(const M& m, IndexType col) { return row_of(transpose(m), col); }

	template <IndexType Column, Matrix M>
	constexpr auto column_of(const M& m) { return RowOf<M, StaticExtent<Column>> { transpose(m) }; }

	template <Matrix L, Matrix R>
	struct KroneckerProduct {
		L left;
		R right;

		constexpr KroneckerProduct(const L& l, const R& r)
			: left(l), right(r)
		{}

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

	template <Matrix L, Matrix R>
	constexpr auto kronecker_product(const L& l, const R& r) { return KroneckerProduct<L, R> { l, r }; };

	template <Matrix L, Matrix R>
	struct MatrixMultiplication {
		L left;
		R right;

		constexpr MatrixMultiplication(const L& l, const R& r) : left(l), right(r) {
			assert_extent(left.column_count(), right.row_count(), std::equal_to<>{});
		}

		constexpr auto operator[] (IndexType row, IndexType column) const {
			auto&& l = row_of(left, row);
			auto&& r = column_of(right, column);
			assert_extent(l.size(), r.size(), std::equal_to<>{});
			using T = decltype(l[0]);
			//dot product
			T sum = static_cast<T>(0);
			for (IndexType i = 0; i < l.size().get(); ++i)
				sum += l[i] * r[i];
			return sum;
		}

		constexpr auto row_count() const { return left.row_count(); }
		constexpr auto column_count() const { return right.column_count(); }
	};

	template <Matrix L, Matrix R>
	constexpr auto operator* (const L& l, const R& r) {
		return MatrixMultiplication<L, R> { l, r };
	}

	template <Matrix L, Matrix R>
	constexpr auto operator/ (const L& l, const R& r) {
		return l * inverse(r);
	}

	template <Matrix L, Matrix R, typename BinaryOperator>
	struct MatrixComponentWiseBinaryOperation {
		L left;
		R right;
    	BinaryOperator op;

		constexpr MatrixComponentWiseBinaryOperation(const L& l, const R& r, const BinaryOperator& op = {})
			: left(l), right(r), op(op)
		{
			assert_extent(left.row_count(), right.row_count(), std::equal_to<>{});
			assert_extent(left.column_count(), right.column_count(), std::equal_to<>{});
		}

		constexpr auto operator[] (IndexType row, IndexType column) const {
			return op(left[row, column], right[row, column]);
		}

		constexpr auto row_count() const { return left.row_count(); }
		constexpr auto column_count() const { return left.column_count(); }
	};

	template <Matrix L, Matrix R>
	constexpr auto operator+ (const L& l, const R& r) {
		return MatrixComponentWiseBinaryOperation<L, R, std::plus<>> { l, r };
	}

	template <Matrix L, Matrix R>
	constexpr auto operator- (const L& l, const R& r) {
		return MatrixComponentWiseBinaryOperation<L, R, std::minus<>> { l, r };
	}

	template <Matrix L, Matrix R>
	constexpr auto hadamard_product(const L& l, const R& r) {
		return MatrixComponentWiseBinaryOperation<L, R, std::multiplies<>> { l, r };
	}

	template <Matrix L, Matrix R>
	constexpr auto hadamard_division(const L& l, const R& r) {
		return MatrixComponentWiseBinaryOperation<L, R, std::divides<>> { l, r };
	}

	template <Matrix L, typename Field, typename BinaryOperator>
	struct MatrixScalarBinaryOperation {
		L left;
		Field right;
    	BinaryOperator op;

		constexpr MatrixScalarBinaryOperation(const L& l, const Field& r, const BinaryOperator& op = {})
			: left(l), right(r), op(op)
		{}

		constexpr auto operator[] (IndexType row, IndexType column) const {
			return op(left[row, column], right);
		}

		constexpr auto row_count() const { return left.row_count(); }
		constexpr auto column_count() const { return left.column_count(); }
	};

	template <Matrix L, typename Field>
	constexpr auto operator* (const L& l, const Field& r) {
		return MatrixScalarBinaryOperation<L, Field, std::multiplies<>> { l, r };
	}
	template <typename Field, Matrix R>
	constexpr auto operator* (const Field& l, const R& r) { return r * l; }

	template <Matrix L, typename Field>
	constexpr auto operator/ (const L& l, const Field& r) {
		return MatrixScalarBinaryOperation<L, Field, std::divides<>> { l, r };
	}

	template <Matrix L, typename Field>
	constexpr auto operator+ (const L& l, const Field& r) {
		return MatrixScalarBinaryOperation<L, Field, std::plus<>> { l, r };
	}
	template <typename Field, Matrix R>
	constexpr auto operator+ (const Field& l, const R& r) { return r + l; }

	template <Matrix L, typename Field>
	constexpr auto operator- (const L& l, const Field& r) {
		return MatrixScalarBinaryOperation<L, Field, std::minus<>> { l, r };
	}

	template <Matrix M>
	constexpr auto min(const M& m) {
		auto minimum = m[0,0];
		for(IndexType row = 0; row < m.row_count().get(); ++row)
			for(IndexType column = 0; column < m.column_count().get(); ++column)
				if(minimum > m[row, column]) minimum = m[row, column];
        return minimum;
	}

	template <Matrix M>
	constexpr auto max(const M& m) {
		auto maximum = m[0,0];
		for(IndexType row = 0; row < m.row_count().get(); ++row)
			for(IndexType column = 0; column < m.column_count().get(); ++column)
				if(maximum < m[row, column]) maximum = m[row, column];
        return maximum;
	}

	template <Matrix M>
	constexpr auto norm_min(const M& m) {
		using std::abs;
		auto norm = abs(m[0,0]);
		for(IndexType row = 0; row < m.row_count().get(); ++row)
			for(IndexType column = 0; column < m.column_count().get(); ++column)
				if(norm > abs(m[row, column])) norm = abs(m[row, column]);
        return norm;
	}

	template <Matrix M>
	constexpr auto norm_max(const M& m) {
		using std::abs;
		auto norm = abs(m[0,0]);
		for(IndexType row = 0; row < m.row_count().get(); ++row)
			for(IndexType column = 0; column < m.column_count().get(); ++column)
				if(norm < abs(m[row, column])) norm = abs(m[row, column]);
        return norm;
	}

	template <Matrix M>
	constexpr auto norm_frobenius(const M& m) {
		using std::sqrt;
		auto sum = static_cast<std::remove_reference_t<decltype(m[0,0])>>(0);
		for(IndexType row = 0; row < m.row_count().get(); ++row)
			for(IndexType column = 0; column < m.column_count().get(); ++column)
				sum += m[row, column] * m[row, column];
        return sqrt(sum);
	}
	template <Matrix M>
	constexpr auto norm_euclidean(const M& m) { return norm_frobenius(m); }
	template <Matrix M>
	constexpr auto norm(const M& m) { return norm_frobenius(m); }

	template <Matrix M>
	constexpr auto normalize_frobenius(const M& m) {
        return m/norm_frobenius(m);
	}
	template <Matrix M>
	constexpr auto normalize_euclidean(const M& m) { return normalize_frobenius(m); }
	template <Matrix M>
	constexpr auto normalize(const M& m) { return normalize_frobenius(m); }

	template <Matrix M>
	constexpr auto normalize_min(const M& m) { return m/std::remove_reference_t<decltype(m[0,0])>{norm_min(m)}; }

	template <Matrix M>
	constexpr auto normalize_max(const M& m) { return m/std::remove_reference_t<decltype(m[0,0])>{norm_max(m)}; }

	template <Matrix M>
	constexpr auto normalize_minmax(const M& m) {
		auto minimum = min(m);
		auto maximum = max(m);
		return (m - minimum)/(maximum - minimum);
	}

	template <typename T, Extent ExtD>
	struct DiscreteFourierTransformMatrix {
		using Field = std::complex<T>;

		ExtD dimension;
		T norm;
		Field omega;

		constexpr DiscreteFourierTransformMatrix(const ExtD& dimension = {})
			: dimension(dimension)
		{
			constexpr Field i = Field(static_cast<T>(0.0), static_cast<T>(1.0));
			constexpr T pi = std::numbers::pi_v<T>;
			
			norm = static_cast<T>(1)/std::sqrt(static_cast<T>(dimension.get()));
			omega = std::exp(static_cast<T>(-2) * pi * i / Field(static_cast<T>(dimension.get())));
		}

		constexpr Field operator[] ([[maybe_unused]] IndexType row, [[maybe_unused]] IndexType column) const {
			return std::pow(omega, static_cast<T>(column*row))*norm;
		}

		constexpr auto row_count() const { return dimension; }
		constexpr auto column_count() const { return dimension; }
	};

	template <IndexType Dimension, typename T = float>
	constexpr auto mat_DFT() {
		return DiscreteFourierTransformMatrix<T, StaticExtent<Dimension>> {};
	}
	template <typename T = float>
	constexpr auto mat_DFT(IndexType dimension) {
		return DiscreteFourierTransformMatrix<T, DynamicExtent> { dimension };
	}

	template <Extent Dimension, typename T = float>
	constexpr auto mat_walsh_sylvester([[maybe_unused]]IndexType dimension) {
		if constexpr (Dimension::is_static()) {
			if constexpr (Dimension::get() == 1) {
				//Hadamard matrix of order 1
				return mat<1, 1, T>({1});
			} else if constexpr (Dimension::get() == 2) {
				//Hadamard matrix of order 2
				return mat<2, 2, T>({
					1,  1,
					1, -1,
				});
			} else {
				//Hadamard matrix of order N
				auto H_2 = mat_walsh_sylvester<StaticExtent<2>, T>(0);
				auto H_n = mat_walsh_sylvester<StaticExtent<Dimension::get()/2>, T>(0);
				return kronecker_product(H_2, H_n);
			}
		} else {
			if (dimension==1) {
				return mat<T>({1}, 1, 1);
			} else if (dimension==2) {
				return mat<T>({ 1,  1, 1, -1, }, 2, 2);
			} else {
				auto H_2 = mat_walsh_sylvester<DynamicExtent, T>(2);
				auto H_n = mat_walsh_sylvester<DynamicExtent, T>(dimension/2);
				mat_dynamic_t<T> ret = kronecker_product(H_2, H_n);
				return ret;
			}
		}
	}

	template <IndexType Dimension, typename T = float>
	constexpr auto mat_walsh_sylvester() {
		static_assert(is_power_of_two(Dimension));
		return mat_walsh_sylvester<StaticExtent<Dimension>, T>(0);
	}

	template <typename T = float>
	inline auto mat_walsh_sylvester(IndexType dimension) {
		assert(is_power_of_two(dimension));
		return mat_walsh_sylvester<DynamicExtent, T>(dimension);
	}

	template <Vector V>
	struct MatrixScaling {
		V coefficients;

		constexpr MatrixScaling(const V& coeffs) : coefficients(coeffs) {}

		constexpr auto operator[] (IndexType row, IndexType column) const {
			return row == column ?
				coefficients[row] :
				static_cast<decltype(coefficients[row])>(0);
		}

		constexpr auto row_count() const { return coefficients.size(); }
		constexpr auto column_count() const { return coefficients.size(); }
	};

	template <Vector V>
	constexpr auto scaling(const V& coefficients) { return MatrixScaling<V> { coefficients }; }

	template <Vector V>
	struct MatrixTranslation {
		V coefficients;

		constexpr MatrixTranslation(const V& coeffs) : coefficients(coeffs) {}

		constexpr auto operator[] (IndexType row, IndexType column) const {
			constexpr auto zero = static_cast<decltype(coefficients[row])>(0);
			constexpr auto one = static_cast<decltype(coefficients[row])>(1);
			return row == column ? one : (column == coefficients.size().get()? coefficients[row] : zero);
		}

		constexpr auto row_count() const { return coefficients.size()+1; }
		constexpr auto column_count() const { return coefficients.size()+1; }
	};

	template <Vector V>
	constexpr auto translation(const V& coefficients) { return MatrixTranslation<V> { coefficients }; }

	template <typename Field, Vector U, Vector V>
	struct MatrixRotation {
		U basis_u;
		V basis_v;
		Field theta;

		MatrixRotation(const V& basis_u, const V& basis_v, Field theta)
			: basis_u(basis_u), basis_v(basis_v), theta(theta)
		{
			assert_extent(basis_u.size(), basis_v.size(), std::equal_to<>{});
		}

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

	template <typename Field, Vector U, Vector V>
	inline auto rotation(const U& basis_u, const V& basis_v, Field theta) {
		return MatrixRotation<Field, U, V> { basis_u, basis_v, theta };
	}

	template <typename T, bool ColumnMajor = false>
	constexpr auto vec_ref(T* base_ptr, IndexType size) {
		return row_of(mat_ref<T, ColumnMajor>(base_ptr, 1, size), 0);
	}

	template <IndexType Size, typename T, bool ColumnMajor = false>
	constexpr auto vec_ref(T* base_ptr) {
		return row_of<0>(mat_ref<1, Size, T, ColumnMajor>(base_ptr));
	}
	
	template <typename T, bool ColumnMajor = false>
	constexpr auto vec_ref(std::initializer_list<T> elements) {
		return vec_ref<const T, ColumnMajor>(std::data(elements), std::size(elements));
	}

	template <typename T, IndexType N, bool ColumnMajor = false>
	constexpr auto vec_ref(const std::span<T, N>& span) {
		if constexpr (N == std::dynamic_extent) {
			return vec_ref<T, ColumnMajor>(span.data(), span.size());
		} else {
			return vec_ref<N, T, ColumnMajor>(span.data());
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

	template <Container T, bool ColumnMajor = false>
	constexpr auto vec_ref(T& container) {
		return vec_ref<typename T::value_type, ColumnMajor>(std::data(container), std::size(container));
	}

	template <Vector V>
	struct AsRowVector {
		V vector;

		constexpr auto operator[] (IndexType row, IndexType column) const {
			assert(row == 0);
			return vector[column];
		}

		constexpr auto row_count() const { return StaticExtent<1>(); }
		constexpr auto column_count() const { return vector.size(); }
	};

	template <Vector V>
	constexpr auto as_row(const V& vector) {
		return AsRowVector<V>{ vector };
	}

	template <Vector V>
	struct AsColumnVector {
		V vector;

		constexpr auto operator[] (IndexType row, IndexType column) const {
			assert(column == 0);
			return vector[row];
		}

		constexpr auto row_count() const { return vector.size(); }
		constexpr auto column_count() const { return StaticExtent<1>(); }
	};

	template <Vector V>
	constexpr auto as_column(const V& vector) {
		return AsColumnVector<V>{ vector };
	}

	template <Vector V, Extent E, bool ColumnMajor = false>
	struct VectorAsMatrix {
		V vector;
		E stride;

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

	template <Vector V, IndexType Stride, bool ColumnMajor = false>
	constexpr auto as_matrix(const V& vector) {
		return VectorAsMatrix<V, StaticExtent<Stride>, ColumnMajor>{ vector, {} };
	}

	template <Vector V, bool ColumnMajor = false>
	constexpr auto as_matrix(const V& vector, IndexType stride) {
		return VectorAsMatrix<V, DynamicExtent, ColumnMajor>{ vector, stride };
	}

	template <Vector A, Vector B>
	constexpr auto inner_product(const A& a, const B& b) {
		return (as_row(a) * as_column(b))[0, 0];
	}
	template <Vector A, Vector B>
	constexpr auto dot_product(const A& a, const B& b) { return inner_product(a, b); }
	template <Vector A, Vector B>
	constexpr auto dot(const A& a, const B& b) { return inner_product(a, b); }

	template <Vector L, Vector R>
	struct OuterProduct {
		L left;
		R right;

		constexpr OuterProduct(const L& l, const R& r)
			: left(l), right(r)
		{}

		constexpr auto operator[] (IndexType row, IndexType column) const {
			return left[row]*right[column];
		}

		constexpr auto row_count() const { return left.size(); }
		constexpr auto column_count() const { return right.size(); }
	};

	template <Vector L, Vector R>
	constexpr auto outer_product(const L& l, const R& r) {
		return OuterProduct{ l, r };
	}

	template <Vector L, Vector R>
	struct CrossProduct {
		L left;
		R right;

		constexpr CrossProduct(const L& l, const R& r)
			: left(l), right(r)
		{
			assert_extent(left.size(), StaticExtent<3>{}, std::equal_to<>{});
			assert_extent(right.size(), StaticExtent<3>{}, std::equal_to<>{});
		}

		constexpr auto operator[] (IndexType i) const {
			switch(i) {
				case 0: return left[1]*right[2] - left[2]*right[1];
				case 1: return left[2]*right[0] - left[0]*right[2];
				case 2: return left[0]*right[1] - left[1]*right[0];
			}
			return static_cast<std::remove_reference_t<decltype(left[0])>>(0);
		}

		constexpr auto size() const { return StaticExtent<3>{}; }
	};

	template <Vector L, Vector R>
	constexpr auto cross_product(const L& l, const R& r) {
		return CrossProduct{ l, r };
	}
	template <Vector L, Vector R>
	constexpr auto cross(const L& l, const R& r) { return cross_product( l, r ); }
	
	template <Vector L, Vector R>
	constexpr auto kronecker_product(const L& l, const R& r) {
		return column_of(kronecker_product( as_column(l), as_column(r) ), 0);
	}
	template <VectorStatic L, VectorStatic R>
	constexpr auto kronecker_product(const L& l, const R& r) {
		return column_of<0>(kronecker_product( as_column(l), as_column(r) ));
	}

	template <Matrix M, VectorStatic V>
	constexpr auto operator* (const M& m, const V& v) { return column_of<0>(m * as_column(v)); }
	template <VectorStatic V, Matrix M>
	constexpr auto operator* (const V& v, const M& m) { return row_of<0>(as_row(v) * m); }

	template <Matrix M, Vector V>
	constexpr auto operator* (const M& m, const V& v) { return column_of(m * as_column(v), 0); }
	template <Vector V, Matrix M>
	constexpr auto operator* (const V& v, const M& m) { return row_of(as_row(v) * m, 0); }

	template <Vector L, Vector R, typename BinaryOperator>
	struct VectorComponentWiseBinaryOperation {
		L left;
		R right;
    	BinaryOperator op;

		constexpr VectorComponentWiseBinaryOperation(const L& l, const R& r, const BinaryOperator& op = {})
			: left(l), right(r), op(op)
		{
			assert_extent(left.size(), right.size(), std::equal_to<>{});
		}

		constexpr auto operator[] (IndexType i) const {
			return op(left[i], right[i]);
		}

		constexpr auto size() const { return left.size(); }
	};

	template <Vector L, Vector R>
	constexpr auto operator* (const L& l, const R& r) { return VectorComponentWiseBinaryOperation<L, R, std::multiplies<>>{ l, r }; }
	template <Vector L, Vector R>
	constexpr auto operator/ (const L& l, const R& r) { return VectorComponentWiseBinaryOperation<L, R, std::divides<>>{ l, r }; }
	template <Vector L, Vector R>
	constexpr auto operator+ (const L& l, const R& r) { return VectorComponentWiseBinaryOperation<L, R, std::plus<>>{ l, r }; }
	template <Vector L, Vector R>
	constexpr auto operator- (const L& l, const R& r) { return VectorComponentWiseBinaryOperation<L, R, std::minus<>>{ l, r }; }

	template <Vector L, typename Field, typename BinaryOperator>
	struct VectorScalarBinaryOperation {
		L left;
		Field right;
    	BinaryOperator op;

		constexpr VectorScalarBinaryOperation(const L& l, const Field& r, const BinaryOperator& op = {})
			: left(l), right(r), op(op)
		{}

		constexpr auto operator[] (IndexType i) const {
			return op(left[i], right);
		}

		constexpr auto size() const { return left.size(); }
	};

	template <Vector V, typename Field>
	constexpr auto operator* (const V& l, const Field& r) { return VectorScalarBinaryOperation<V, Field, std::multiplies<>>{ l, r }; }
	template <typename Field, Vector V>
	constexpr auto operator* (const Field& l, const V& r) { return VectorScalarBinaryOperation<V, Field, std::multiplies<>>{ r, l }; }
	template <Vector V, typename Field>
	constexpr auto operator/ (const V& l, const Field& r) { return VectorScalarBinaryOperation<V, Field, std::divides<>>{ l, r }; }
	template <Vector V, typename Field>
	constexpr auto operator+ (const V& l, const Field& r) { return VectorScalarBinaryOperation<V, Field, std::plus<>>{ l, r }; }
	template <Vector V, typename Field>
	constexpr auto operator- (const V& l, const Field& r) { return VectorScalarBinaryOperation<V, Field, std::minus<>>{ l, r }; }

	template <Vector V>
	constexpr auto min(const V& v) {
		auto minimum = v[0];
		for(IndexType i = 0; i < v.size().get(); ++i)
			if(minimum > v[i]) minimum = v[i];
        return minimum;
	}

	template <Vector V>
	constexpr auto max(const V& v) {
		auto maximum = v[0];
		for(IndexType i = 0; i < v.size().get(); ++i)
			if(maximum < v[i]) maximum = v[i];
        return maximum;
	}

	template <Vector V>
	constexpr auto norm_min(const V& v) {
		using std::abs;
		auto norm = abs(v[0]);
		for(IndexType i = 0; i < v.size().get(); ++i)
			if(norm > abs(v[i])) norm = abs(v[i]);
        return norm;
	}

	template <Vector V>
	constexpr auto norm_max(const V& v) {
		using std::abs;
		auto norm = abs(v[0]);
		for(IndexType i = 0; i < v.size().get(); ++i)
			if(norm < abs(v[i])) norm = abs(v[i]);
        return norm;
	}

	template <Vector V>
	constexpr auto norm_frobenius(const V& v) {
		using std::sqrt;
        return sqrt(dot(v, v));
	}
	template <Vector V>
	constexpr auto norm_euclidean(const V& v) { return norm_frobenius(v); }
	template <Vector V>
	constexpr auto norm(const V& v) { return norm_frobenius(v); }

	template <Vector V>
	constexpr auto normalize_frobenius(const V& v) {
        return v/norm_frobenius(v);
	}
	template <Vector V>
	constexpr auto normalize_euclidean(const V& v) { return normalize_frobenius(v); }
	template <Vector V>
	constexpr auto normalize(const V& v) { return normalize_frobenius(v); }

	template <Vector V>
	constexpr auto normalize_min(const V& v) { return v/std::remove_reference_t<decltype(v[0])>{norm_min(v)}; }

	template <Vector V>
	constexpr auto normalize_max(const V& v) { return v/std::remove_reference_t<decltype(v[0])>{norm_max(v)}; }

	template <Vector V>
	constexpr auto normalize_minmax(const V& v) {
		auto minimum = min(v);
		auto maximum = max(v);
		return (v - minimum)/(maximum - minimum);
	}

	template <Matrix M>
	inline void print(const M& mat, std::ostream& os = std::cout, std::streamsize spacing_width = 12) {
		for (IndexType row = 0; row < mat.row_count().get(); ++row) {
			for (IndexType col = 0; col < mat.column_count().get(); ++col)
				os << std::setw(spacing_width) << mat[row, col] << ",";
			os << std::endl;
		}
	}

	template <Vector V>
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
