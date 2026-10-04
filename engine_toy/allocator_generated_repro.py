
def bounded_solve(operator, target, lower, upper, initial, options, linear_term, x, hessian, drive, gradient, blocked, matrix, rhs, current_row, pivot_row_values, current_rhs, pivot_rhs, pivot_rows, pivot_magnitudes, telemetry):
    for initialize_column in range(25):
        x[initialize_column] = min(upper[initialize_column], max(lower[initialize_column], initial[initialize_column]))
        drive_total = -linear_term[initialize_column]
        for drive_row in range(6):
            drive_total = drive_total + operator[drive_row * 25 + initialize_column] * target[drive_row]
        drive[initialize_column] = drive_total
        for hessian_column in range(25):
            hessian_total = options[0] if initialize_column == hessian_column else 0.0
            for hessian_row in range(6):
                hessian_total = hessian_total + operator[hessian_row * 25 + initialize_column] * operator[hessian_row * 25 + hessian_column]
            hessian[initialize_column * 25 + hessian_column] = hessian_total
    telemetry[0] = 0.0
    for iteration in range(128):
        projected_norm = 0.0
        for gradient_column in range(25):
            gradient_total = -drive[gradient_column]
            for gradient_inner in range(25):
                gradient_total = gradient_total + hessian[gradient_column * 25 + gradient_inner] * x[gradient_inner]
            gradient[gradient_column] = gradient_total
            at_lower = x[gradient_column] <= lower[gradient_column] and gradient_total >= 0.0
            at_upper = x[gradient_column] >= upper[gradient_column] and gradient_total <= 0.0
            is_blocked = lower[gradient_column] == upper[gradient_column] or at_lower or at_upper
            blocked[gradient_column] = 1.0 if is_blocked else 0.0
            projected_component = 0.0 if is_blocked else abs(gradient_total)
            projected_norm = max(projected_norm, projected_component)
        frozen = projected_norm <= options[1]
        telemetry[0] = telemetry[0] if frozen else (iteration + 1) * 1.0
        for reduced_row in range(25):
            row_blocked = frozen or blocked[reduced_row] > 0.0
            rhs[reduced_row] = 0.0 if row_blocked else -gradient[reduced_row]
            for reduced_column in range(25):
                column_blocked = frozen or blocked[reduced_column] > 0.0
                unit_entry = 1.0 if reduced_row == reduced_column else 0.0
                matrix[reduced_row * 25 + reduced_column] = unit_entry if row_blocked or column_blocked else hessian[reduced_row * 25 + reduced_column]
        for lane in range(1):
            matrix_base = lane * 25 * 25
            vector_base = lane * 25
            for pivot_column in range(25):
                pivot_state_base = matrix_base + pivot_column * 25
                pivot_rows[pivot_state_base + pivot_column] = pivot_column * 1.0
                pivot_magnitudes[pivot_state_base + pivot_column] = abs(matrix[matrix_base + pivot_column * 25 + pivot_column])
                for candidate_row in range(pivot_column + 1, 25):
                    candidate_magnitude = abs(matrix[matrix_base + candidate_row * 25 + pivot_column])
                    prior_state = pivot_state_base + candidate_row - 1
                    candidate_state = pivot_state_base + candidate_row
                    take_candidate = candidate_magnitude > pivot_magnitudes[prior_state]
                    candidate_row_float = candidate_row * 1.0
                    pivot_rows[candidate_state] = candidate_row_float if take_candidate else pivot_rows[prior_state]
                    pivot_magnitudes[candidate_state] = candidate_magnitude if take_candidate else pivot_magnitudes[prior_state]
                pivot_row = pivot_rows[pivot_state_base + 25 - 1]
                for swap_column in range(25):
                    pivot_index = matrix_base + pivot_column * 25 + swap_column
                    swap_index = matrix_base + pivot_row * 25 + swap_column
                    current_row[vector_base + swap_column] = matrix[pivot_index]
                    pivot_row_values[vector_base + swap_column] = matrix[swap_index]
                for swap_column in range(25):
                    pivot_index = matrix_base + pivot_column * 25 + swap_column
                    swap_index = matrix_base + pivot_row * 25 + swap_column
                    matrix[pivot_index] = pivot_row_values[vector_base + swap_column]
                    matrix[swap_index] = current_row[vector_base + swap_column]
                rhs_pivot_index = vector_base + pivot_column
                rhs_swap_index = vector_base + pivot_row
                current_rhs[rhs_pivot_index] = rhs[rhs_pivot_index]
                pivot_rhs[rhs_pivot_index] = rhs[rhs_swap_index]
                rhs[rhs_pivot_index] = pivot_rhs[rhs_pivot_index]
                rhs[rhs_swap_index] = current_rhs[rhs_pivot_index]
                pivot_value = matrix[matrix_base + pivot_column * 25 + pivot_column]
                for normalization_column in range(pivot_column, 25):
                    normalization_index = matrix_base + pivot_column * 25 + normalization_column
                    matrix[normalization_index] = matrix[normalization_index] / pivot_value
                rhs[vector_base + pivot_column] = rhs[vector_base + pivot_column] / pivot_value
                for elimination_row in range(pivot_column + 1, 25):
                    factor = matrix[matrix_base + elimination_row * 25 + pivot_column]
                    for elimination_column in range(pivot_column, 25):
                        target_index = matrix_base + elimination_row * 25 + elimination_column
                        source_index = matrix_base + pivot_column * 25 + elimination_column
                        matrix[target_index] = matrix[target_index] - factor * matrix[source_index]
                    rhs[vector_base + elimination_row] = rhs[vector_base + elimination_row] - factor * rhs[vector_base + pivot_column]
                for elimination_row in range(25):
                    eliminate_above = elimination_row < pivot_column
                    factor = matrix[matrix_base + elimination_row * 25 + pivot_column] if eliminate_above else 0.0
                    for elimination_column in range(pivot_column, 25):
                        target_index = matrix_base + elimination_row * 25 + elimination_column
                        source_index = matrix_base + pivot_column * 25 + elimination_column
                        matrix[target_index] = matrix[target_index] - factor * matrix[source_index]
                    rhs[vector_base + elimination_row] = rhs[vector_base + elimination_row] - factor * rhs[vector_base + pivot_column]

        step_fraction = 1.0
        for fraction_column in range(25):
            positive_fraction = (upper[fraction_column] - x[fraction_column]) / rhs[fraction_column] if rhs[fraction_column] > 0.0 else 1.0
            negative_fraction = (lower[fraction_column] - x[fraction_column]) / rhs[fraction_column] if rhs[fraction_column] < 0.0 else 1.0
            step_fraction = min(step_fraction, positive_fraction, negative_fraction)
        # A released bound can oppose the coupled Newton direction. Its
        # zero feasible step uses the projected steepest-descent direction;
        # exact quadratic line minimization keeps the same objective.
        use_projected = step_fraction <= 0.0
        for safeguard_column in range(25):
            descent_component = -gradient[safeguard_column] if blocked[safeguard_column] == 0.0 else 0.0
            rhs[safeguard_column] = descent_component if use_projected else rhs[safeguard_column]
        descent_product = 0.0
        curvature = 0.0
        step_fraction = 1.0
        for search_column in range(25):
            descent_product = descent_product + gradient[search_column] * rhs[search_column]
            curvature_row = 0.0
            for curvature_column in range(25):
                curvature_row = curvature_row + hessian[search_column * 25 + curvature_column] * rhs[curvature_column]
            curvature = curvature + rhs[search_column] * curvature_row
            search_positive = (upper[search_column] - x[search_column]) / rhs[search_column] if rhs[search_column] > 0.0 else 1.0
            search_negative = (lower[search_column] - x[search_column]) / rhs[search_column] if rhs[search_column] < 0.0 else 1.0
            step_fraction = min(step_fraction, search_positive, search_negative)
        minimum_fraction = -descent_product / curvature if curvature > 0.0 else 1.0
        step_fraction = max(0.0, min(step_fraction, minimum_fraction))
        for update_column in range(25):
            x[update_column] = min(upper[update_column], max(lower[update_column], x[update_column] + step_fraction * rhs[update_column]))
    final_norm = 0.0
    for final_column in range(25):
        final_gradient = -drive[final_column]
        for final_inner in range(25):
            final_gradient = final_gradient + hessian[final_column * 25 + final_inner] * x[final_inner]
        final_lower = x[final_column] <= lower[final_column] and final_gradient >= 0.0
        final_upper = x[final_column] >= upper[final_column] and final_gradient <= 0.0
        final_blocked = lower[final_column] == upper[final_column] or final_lower or final_upper
        final_component = 0.0 if final_blocked else abs(final_gradient)
        final_norm = max(final_norm, final_component)
    telemetry[1] = final_norm
    telemetry[2] = 1.0 if final_norm <= options[1] else 0.0
    return x, telemetry
