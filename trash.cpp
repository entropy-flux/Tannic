
/*
    /**
     * @brief Assigns values to a 1D tensor from an initializer list.
     *
     * @tparam T Element type of the initializer list.
     * @param values A list of values to assign to the tensor.
     *
     * @details
     * This operator assigns values to an **existing 1D tensor**:
     *
     * ```cpp
     * Tensor X(float32, {3});
     * X = {1, 2, 3};  // values cast to float32
     * ```
     *
     * Requirements:
     * - The tensor must already be initialized.
     * - The tensor must be rank-1.
     * - The tensor must be contiguous.
     * - The size of `values` must match the tensor shape.
     *
     * Type conversion:
     * - Values are cast to the tensor’s existing dtype before being written.
     * - This ensures that the tensor’s dtype does not change after assignment.
     */ 
    template<typename T>
    Tensor& operator=(std::initializer_list<T> const& values) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");
    
        if (!is_initialized()) 
            initialize();

        if (rank() != 1 || shape_[0] != values.size())
            throw Exception("Shape mismatch in assignment from initializer_list");
 
        if (dtype_ == boolean) {
            std::ptrdiff_t index = 0;
            for (auto const& value : values) {
                assign((bool const*)(&value), index);
                ++index;
            }
        } 
        
        else { 
            auto fill = [this, &values](auto cast) { 
                using Cast = decltype(cast);
                size_t index = 0;
                for (auto value : values) {
                    Cast casted = value;
                    assign(expression::tobytes(casted), index * dsizeof(dtype_));
                    ++index;
                }
            }; 

            switch (dtype_) {
                case int8:    fill(int8_t{});   break;
                case int16:   fill(int16_t{});  break;
                case int32:   fill(int32_t{});  break;
                case int64:   fill(int64_t{});  break;
                case float32: fill(float{});    break;
                case float64: fill(double{});   break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        }
        
        return *this;
  
    }

    /**
     * @brief Assigns values to a 2D tensor from a nested initializer list.
     *
     * @tparam T Element type of the nested initializer list.
     * @param values A nested list of values to assign to the tensor (rows).
     *
     * @details
     * This operator assigns values to an **existing 2D tensor**:
     *
     * ```cpp
     * Tensor Y(float32, {2, 3});
     * Y = {
     *     {1, 2, 3},
     *     {4, 5, 6}
     * };  // values cast to float32
     * ```
     *
     * Requirements:
     * - The tensor must already be initialized.
     * - The tensor must be rank-2.
     * - The tensor must be contiguous.
     * - All rows must have the same length, matching the tensor’s second dimension.
     *
     * Type conversion:
     * - Values are cast to the tensor’s existing dtype before being written.
     */
    template<typename T>
    Tensor& operator=(std::initializer_list<std::initializer_list<T>> const& values) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");
            
        if (!is_initialized()) 
            initialize();

        if (rank() != 2 || shape_[0] != values.size() || shape_[1] != values.begin()->size())
            throw Exception("Shape mismatch in assignment from nested initializer_list"); 

        if (dtype_ == boolean) {
            std::ptrdiff_t index = 0;
            for (auto const& row : values) {
                if (row.size() != shape_[1])
                    throw Exception("Row length mismatch in assignment from initializer_list");
                for (auto const& value : row) {
                    assign((bool const*)(&value), index);
                    ++index;
                }
            } 
        }

        else { 
            auto fill = [this, &values](auto cast) { 
                using Cast = decltype(cast);
                size_t index = 0;
                for (auto const& row : values) {
                    if (row.size() != shape_[1])
                        throw Exception("Row length mismatch in assignment from initializer_list");

                    for (auto value : row) {
                        Cast casted = value;
                        assign(expression::tobytes(casted), index * dsizeof(dtype_));
                        ++index;
                    }
                }
            };

            switch (dtype_) {
                case int8:    fill(int8_t{});   break;
                case int16:   fill(int16_t{});  break;
                case int32:   fill(int32_t{});  break;
                case int64:   fill(int64_t{});  break;
                case float32: fill(float{});    break;
                case float64: fill(double{});   break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        } 

        return *this;
    }

    /**
     * @brief Assigns values to a 4D tensor from a quadruple-nested initializer list.
     *
     * @tparam T Element type of the nested initializer list.
     * @param values A quadruple-nested list of values to assign to the tensor.
     *
     * @details
     * This operator assigns values to an **existing 4D tensor**:
     *
     * ```cpp
     * Tensor W(float32, {2, 2, 2, 2});  
     * W = {
     *     {
     *         {
     *             {1,  2}, {3,  4}
     *         },
     *         {
     *             {5,  6}, {7,  8}
     *         }
     *     },
     *     {
     *         {
     *             {9, 10}, {11, 12}
     *         },
     *         {
     *             {13, 14}, {15, 16}
     *         }
     *     }
     * };  // values cast to float32
     * ```
     *
     * Requirements:
     * - The tensor must already be initialized.
     * - The tensor must be rank-4.
     * - The tensor must be contiguous.
     * - All nested dimensions must match the tensor’s shape.
     *
     * Type conversion:
     * - Values are cast to the tensor’s existing dtype before being written.
     */
    template<typename T>
    Tensor& operator=(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> const& values) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");

        if (!is_initialized()) 
            initialize();

        if (rank() != 4 
            || shape_[0] != values.size() 
            || shape_[1] != values.begin()->size() 
            || shape_[2] != values.begin()->begin()->size() 
            || shape_[3] != values.begin()->begin()->begin()->size())
            throw Exception("Shape mismatch in assignment from quadruple-nested initializer_list");


        if (dtype_ == boolean) {
            std::ptrdiff_t index = 0;
            for (auto const& tensor3D : values) {
                if (tensor3D.size() != shape_[1])
                    throw Exception("3D tensor count mismatch");

                for (auto const& matrix : tensor3D) {
                    if (matrix.size() != shape_[2])
                        throw Exception("Matrix row count mismatch");

                    for (auto const& row : matrix) {
                        if (row.size() != shape_[3])
                            throw Exception("Row length mismatch");

                        for (auto const& value : row) {
                            assign((bool const*)(&value), index);
                            ++index;
                        }
                    }
                }
            }
            return *this;
        }

        else { 
            auto fill = [this, &values](auto cast) { 
                using Cast = decltype(cast);
                size_t index = 0;
                for (auto const& tensor3D : values) {
                    if (tensor3D.size() != shape_[1])
                        throw Exception("3D tensor count mismatch");

                    for (auto const& matrix : tensor3D) {
                        if (matrix.size() != shape_[2])
                            throw Exception("Matrix row count mismatch");

                        for (auto const& row : matrix) {
                            if (row.size() != shape_[3])
                                throw Exception("Row length mismatch");

                            for (auto value : row) {
                                Cast casted = value;
                                assign(expression::tobytes(casted), index * dsizeof(dtype_));
                                ++index;
                            }
                        }
                    }
                }
            };

            switch (dtype_) {
                case int8:    fill(int8_t{});   break;
                case int16:   fill(int16_t{});  break;
                case int32:   fill(int32_t{});  break;
                case int64:   fill(int64_t{});  break;
                case float32: fill(float{});    break;
                case float64: fill(double{});   break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        } 
        return *this;
    }

*/