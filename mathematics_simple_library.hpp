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
	constexpr T kronecker_delta(IndexType i, IndexType j) {
		return static_cast<T>(i==j);
	}

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
	
	template <typename V>
	concept Vector = requires(const V& vector, IndexType element) {
		{ vector[element] };
		{ vector.size() } -> Extent;
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
		constexpr void operator= (const M& m) const {
			assert_extent(m.row_count(), this->row_count(), std::equal_to<>{});
			assert_extent(m.column_count(), this->column_count(), std::equal_to<>{});
			for (IndexType row = 0; row < this->row_count().get(); ++row) {
				for (IndexType column = 0; column < this->column_count().get(); ++column) {
					(*this)[row, column] = m[row, column];
				}
			}
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
		return mat_ref<T, ColumnMajor>(std::data(container), rows, columns);
	}
	
	template <Matrix M>
	struct Transpose {
		M matrix;
		constexpr auto operator[] (IndexType row, IndexType column) const { return matrix[column, row]; }
		constexpr auto row_count() const { return matrix.column_count(); }
		constexpr auto column_count() const { return matrix.row_count(); }
	};

	template <Matrix M>
	constexpr auto transpose(const M& m) { return Transpose<M> { m }; }

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
			IndexType row_offset = 0, column_offset = 0;
			for(auto&& r : erased_rows) if(r <= row) ++row_offset;
			for(auto&& c : erased_columns) if(c <= column) ++column_offset;
			row += row_offset; column += column_offset;
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
			if constexpr (Rows::get() == 1){
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
			auto result = zero;
			if(m.row_count().get() == 1) return determinant<StaticExtent<1>, StaticExtent<1>>(m);
			else if(m.row_count().get() == 2) return determinant<StaticExtent<2>, StaticExtent<2>>(m);
			else {
				for (IndexType n = 0; n < m.column_count().get(); n++)
					result +=
						m[0, n] * 
						static_cast<value_type>(n&1?-1:1) * 
						determinant<DynamicExtent, DynamicExtent>(submatrix(m, 0, n));
			}
			return result;
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
	
	template <Matrix M, Extent E>
	struct RowOf {
		M matrix;
		E row;

		RowOf(const M& matrix, const E& row = {}) : matrix(matrix), row(row) {
			assert_extent(row, matrix.row_count(), std::less<>{});
		}

		auto operator[] (IndexType element) const { return matrix[row.get(), element]; }

		auto size() const { return matrix.column_count(); }
	};

	template <Matrix M>
	inline auto row_of(const M& m, IndexType row) { return RowOf<M, DynamicExtent> { m, row }; }
	
	template <IndexType Row, Matrix M>
	inline auto row_of(const M& m) { return RowOf<M, StaticExtent<Row>> { m }; }

	template <Matrix M>
	inline auto column_of(const M& m, IndexType col) { return row_of(transpose(m), col); }

	template <IndexType Column, Matrix M>
	inline auto column_of(const M& m) { return RowOf<M, StaticExtent<Column>> { transpose(m) }; }

	template <Matrix L, Matrix R>
	struct MatrixMultiplication {
		L left;
		R right;

		MatrixMultiplication(const L& l, const R& r) : left(l), right(r) {
			assert_extent(left.column_count(), right.row_count(), std::equal_to<>{});
		}

		auto operator[] (IndexType row, IndexType column) const {
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

		auto row_count() const { return left.row_count(); }
		auto column_count() const { return right.column_count(); }
	};

	template <Matrix L, Matrix R>
	inline auto operator* (const L& l, const R& r) {
		return MatrixMultiplication<L, R> { l, r };
	}

	template <Matrix L, Matrix R, typename BinaryOperator>
	struct MatrixComponentWiseBinaryOperation {
		L left;
		R right;
    	BinaryOperator op;

		MatrixComponentWiseBinaryOperation(const L& l, const R& r, const BinaryOperator& op = {})
			: left(l), right(r), op(op)
		{
			assert_extent(left.row_count(), right.row_count(), std::equal_to<>{});
			assert_extent(left.column_count(), right.column_count(), std::equal_to<>{});
		}

		auto operator[] (IndexType row, IndexType column) const {
			return op(left[row, column], right[row, column]);
		}

		auto row_count() const { return left.row_count(); }
		auto column_count() const { return left.column_count(); }
	};

	template <Matrix L, Matrix R>
	inline auto operator+ (const L& l, const R& r) {
		return MatrixComponentWiseBinaryOperation<L, R, std::plus<>> { l, r };
	}

	template <Matrix L, Matrix R>
	inline auto operator- (const L& l, const R& r) {
		return MatrixComponentWiseBinaryOperation<L, R, std::minus<>> { l, r };
	}

	template <Matrix L, Matrix R>
	inline auto hadamard_product(const L& l, const R& r) {
		return MatrixComponentWiseBinaryOperation<L, R, std::multiplies<>> { l, r };
	}

	template <Matrix L, Matrix R>
	inline auto hadamard_division(const L& l, const R& r) {
		return MatrixComponentWiseBinaryOperation<L, R, std::divides<>> { l, r };
	}

	template <Matrix L, typename Field, typename BinaryOperator>
	struct MatrixScalarBinaryOperation {
		L left;
		Field right;
    	BinaryOperator op;

		MatrixScalarBinaryOperation(const L& l, const Field& r, const BinaryOperator& op = {})
			: left(l), right(r), op(op)
		{}

		auto operator[] (IndexType row, IndexType column) const {
			return op(left[row, column], right);
		}

		auto row_count() const { return left.row_count(); }
		auto column_count() const { return left.column_count(); }
	};

	template <Matrix L, typename Field>
	inline auto operator* (const L& l, const Field& r) {
		return MatrixScalarBinaryOperation<L, Field, std::multiplies<>> { l, r };
	}
	template <typename Field, Matrix R>
	inline auto operator* (const Field& l, const R& r) { return r * l; }

	template <Matrix L, typename Field>
	inline auto operator/ (const L& l, const Field& r) {
		return MatrixScalarBinaryOperation<L, Field, std::divides<>> { l, r };
	}

	template <Matrix L, typename Field>
	inline auto operator+ (const L& l, const Field& r) {
		return MatrixScalarBinaryOperation<L, Field, std::plus<>> { l, r };
	}
	template <typename Field, Matrix R>
	inline auto operator+ (const Field& l, const R& r) { return r + l; }

	template <Matrix L, typename Field>
	inline auto operator- (const L& l, const Field& r) {
		return MatrixScalarBinaryOperation<L, Field, std::minus<>> { l, r };
	}

	template <Vector V>
	struct MatrixScaling {
		V coefficients;

		MatrixScaling(const V& coeffs) : coefficients(coeffs) {}

		auto operator[] (IndexType row, IndexType column) const {
			return row == column ?
				coefficients[row] :
				static_cast<decltype(coefficients[row])>(0);
		}

		auto row_count() const { return coefficients.size(); }
		auto column_count() const { return coefficients.size(); }
	};

	template <Vector V>
	inline auto scaling(const V& coefficients) { return MatrixScaling<V> { coefficients }; }

	template <Vector V>
	struct MatrixTranslation {
		V coefficients;

		MatrixTranslation(const V& coeffs) : coefficients(coeffs) {}

		auto operator[] (IndexType row, IndexType column) const {
			constexpr auto zero = static_cast<decltype(coefficients[row])>(0);
			constexpr auto one = static_cast<decltype(coefficients[row])>(1);
			return row == column ? one : (column == coefficients.size().get()? coefficients[row] : zero);
		}

		auto row_count() const { return coefficients.size()+1; }
		auto column_count() const { return coefficients.size()+1; }
	};

	template <Vector V>
	inline auto translation(const V& coefficients) { return MatrixTranslation<V> { coefficients }; }

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
	inline auto vec_ref(T* base_ptr, IndexType size) {
		return row_of(mat_ref<T, ColumnMajor>(base_ptr, 1, size), 0);
	}

	template <IndexType Size, typename T, bool ColumnMajor = false>
	inline auto vec_ref(T* base_ptr) {
		return row_of<0>(mat_ref<1, Size, T, ColumnMajor>(base_ptr));
	}
	
	template <typename T, bool ColumnMajor = false>
	inline auto vec_ref(std::initializer_list<T> elements) {
		return vec_ref<const T, ColumnMajor>(std::data(elements), std::size(elements));
	}

	template <typename T, IndexType N, bool ColumnMajor = false>
	inline auto vec_ref(const std::span<T, N>& span) {
		if constexpr (N == std::dynamic_extent) {
			return vec_ref<T, ColumnMajor>(span.data(), span.size());
		} else {
			return vec_ref<N, T, ColumnMajor>(span.data());
		}
	}

	template <typename T, IndexType N>
	inline auto vec_ref(const std::array<T, N>& array) {
		return vec_ref(std::span { array });
	}

	template <typename T>
	inline auto vec_ref(const std::vector<T>& vector) {
		return vec_ref(std::span { vector });
	}

	template <Container T, bool ColumnMajor = false>
	inline auto vec_ref(T& container) {
		return vec_ref<T, ColumnMajor>(std::data(container), std::size(container));
	}

	template <Vector V>
	struct AsRowVector {
		V vector;

		auto operator[] (IndexType row, IndexType column) const {
			assert(row == 0);
			return vector[column];
		}

		auto row_count() const { return StaticExtent<1>(); }
		auto column_count() const { return vector.size(); }
	};

	template <Vector V>
	inline auto as_row(const V& vector) {
		return AsRowVector<V>{ vector };
	}

	template <Vector V>
	struct AsColumnVector {
		V vector;

		auto operator[] (IndexType row, IndexType column) const {
			assert(column == 0);
			return vector[row];
		}

		auto row_count() const { return vector.size(); }
		auto column_count() const { return StaticExtent<1>(); }
	};

	template <Vector V>
	inline auto as_column(const V& vector) {
		return AsColumnVector<V>{ vector };
	}

	template <Vector A, Vector B>
	inline auto dot(A a, B b) {
		return (as_row(a) * as_column(b))[0, 0];
	}

	//TODO: improve, specifically the array check part, check instead for dynamic and static container concepts (maybe)
	template <Container T, Extent ExtR, Extent ExtC, bool ColumnMajor>
	struct MatrixObject {
		T data;
		MatrixRef<typename T::value_type, ExtR, ExtC, ColumnMajor> ref;

		template<Matrix M>
		requires (!(ExtR::is_static() && ExtC::is_static()))
		MatrixObject(const M& m) : ref{std::data(data), ExtR{0}, ExtC{0}} {
			(*this) = m;
		}

		template<Matrix M>
		requires (ExtR::is_static() && ExtC::is_static())
		MatrixObject(const M& m) : ref{std::data(data), ExtR{ExtR::get()}, ExtC{ExtC::get()}} {
			(*this) = m;
		}

		MatrixObject(
			const ExtR& rows = {},
			const ExtC& columns = {}
		) : ref{std::data(data), rows, columns} {
			if constexpr(!is_array_v<T>) {
				data.resize(rows.get()*columns.get());
				ref = {std::data(data), rows, columns};
			}
			for(IndexType i = 0; i < rows.get(); ++i)
				for(IndexType j = 0; j < columns.get(); ++j)
					(*this)[i,j] = kronecker_delta<typename T::value_type>(i, j);
		}

		MatrixObject(
			std::initializer_list<typename T::value_type> elements,
			const ExtR& rows = {},
			const ExtC& columns = {}
		) : ref{std::data(data), rows, columns} {
			if constexpr(is_array_v<T>) {
				std::copy_n(elements.begin(), elements.size(), data.begin());
			} else {
				data = elements;
				ref = {std::data(data), rows, columns};
			}
		}

		template <Container C>
		MatrixObject(
			const C& elements,
			const ExtR& rows = {},
			const ExtC& columns = {}
		) : ref{std::data(data), rows, columns} {
			if constexpr(is_array_v<T>) {
				std::copy_n(elements.begin(), elements.size(), data.begin());
			} else {
				data = elements;
				ref = {std::data(data), rows, columns};
			}
		}

		template <Matrix M>
		requires (
			(ExtR::is_static() && ExtC::is_static()) ||
			(!ExtR::is_static() && !ExtC::is_static())
		)
		void operator= (const M& m) {
			if constexpr(!ExtR::is_static() && !ExtC::is_static()) {
				resize(m.row_count().get(), m.column_count().get());
			}
			for(IndexType row = 0; row < ref.row_count().get(); ++row)
				for(IndexType column = 0; column < ref.column_count().get(); ++column)
					(*this)[row, column] = m[row, column];
		}
		
		T::value_type& operator[] (IndexType row, IndexType column) { return ref[row, column]; }
		const T::value_type& operator[] (IndexType row, IndexType column) const { return ref[row, column]; }

		auto row_count() const { return ref.row_count(); }
		auto column_count() const { return ref.column_count(); }

		template<typename Filler = typename T::value_type> requires (!ExtR::is_static() && !ExtC::is_static())
		void resize(IndexType rows, IndexType columns, const Filler& filler_value = static_cast<Filler>(0)) {
			T data_old = data;
			decltype(ref) ref_old = {std::data(data_old), ref.row_count(), ref.column_count()};
			IndexType copy_rows = std::min(ref.row_count().get(), rows);
			IndexType copy_columns = std::min(ref.column_count().get(), columns);
			data.resize(rows*columns);
			ref = {std::data(data), rows, columns};
			for(IndexType row = 0; row < copy_rows; ++row) {
				for(IndexType column = 0; column < copy_columns; ++column)
					ref[row, column] = ref_old[row, column];
				for(IndexType column = copy_columns; column < columns; ++column)
					ref[row, column] = filler_value;
			}
			for(IndexType row = copy_rows; row < rows; ++row) {
				for(IndexType column = 0; column < columns; ++column)
					ref[row, column] = filler_value;
			}
		}
	};

	template <IndexType Rows, IndexType Columns, typename T = float, bool ColumnMajor = false>
	inline auto mat(std::initializer_list<T> elements) {
		return MatrixObject<std::array<T, Rows*Columns>, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor> { elements, {}, {} };
	}

	template <IndexType Rows, IndexType Columns, Container T, bool ColumnMajor = false>
	inline auto mat(const T& container) {
		return MatrixObject<std::array<typename T::value_type, Rows*Columns>, StaticExtent<Rows>, StaticExtent<Columns>, ColumnMajor> { container, {}, {} };
	}

	template <typename T, bool ColumnMajor = false>
	inline auto mat(std::initializer_list<T> elements, IndexType rows, IndexType columns) {
		return MatrixObject<std::vector<T>, DynamicExtent, DynamicExtent, ColumnMajor> { elements, rows, columns };
	}

	template <Container T, bool ColumnMajor = false>
	inline auto mat(const T& container, IndexType rows, IndexType columns) {
		return MatrixObject<std::vector<typename T::value_type>, DynamicExtent, DynamicExtent, ColumnMajor> { container, rows, columns };
	}

	template <IndexType Rows, IndexType Columns, Container T, bool ColumnMajor = false>
	using mat_static_container_t = decltype(mat<Rows, Columns, T, ColumnMajor>(T{}));

	template <IndexType Rows, IndexType Columns, typename T, bool ColumnMajor = false>
	using mat_static_t = decltype(mat<Rows, Columns, T, ColumnMajor>(std::initializer_list<T>{}));

	template <Container T, bool ColumnMajor = false>
	using mat_dynamic_container_t = decltype(mat<T, ColumnMajor>(T{}, 0, 0));

	template <typename T, bool ColumnMajor = false>
	using mat_dynamic_t = decltype(mat<T, ColumnMajor>(std::initializer_list<T>{}, 0, 0));

	template <Matrix M>
	inline void print(const M& mat, std::ostream& os = std::cout, std::streamsize spacing_width = 12) {
		for (IndexType row = 0; row < mat.row_count().get(); ++row) {
			for (IndexType col = 0; col < mat.column_count().get(); ++col)
				os << std::setw(spacing_width) << mat[row, col] << ",";
			os << std::endl;
		}
	}

} // namespace Maths
