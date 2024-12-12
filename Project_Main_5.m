%% gaussnewton
function [x, N_eval, N_iter, normg] = gaussnewton(phi, t, y, x0, tol, printout, plotout)
    tic; % Start timing

    % Inputs:
    % phi      - Function handle for the model function phi(x, t)
    % t        - Vector of independent variable data
    % y        - Vector of dependent variable data (observations)
    % x0       - Initial guess for the parameter vector x
    % tol      - Tolerance for stopping criteria
    % printout - Printing intermediate results (1: print, 0: no print)
    % plotout  - Plotting results (1: plot, 0: no plot)
    
    % Outputs:
    % x        - Final parameter estimate
    % N_eval   - Number of function evaluations
    % N_iter   - Number of iterations performed
    % normg    - Norm of the gradient at the final parameter estimate

    % 1. Initialize variables
    x = x0;         % Start with the initial guess
    N_eval = 0;     % Initialize function evaluation counter
    N_iter = 0;     % Initialize iteration counter
    max_iter = 100; % Set a maximum number of iterations to prevent infinite loops
    
    % 2. Define residual and objective function
    r = @(x) phi(x, t) - y; % Residual function: r(x) = phi(x, t) - y
    f = @(x) sum(r(x).^2);  % Objective function: f(x) = ||r(x)||^2

    % 3. Gauss-Newton loop
    while N_iter < max_iter
        N_iter = N_iter + 1; % Increment the iteration counter
        
        % 3.1. Evaluate the residual vector r(x)
        residual = r(x);
        N_eval = N_eval + 1;  % Increment function evaluation count
        
        % 3.2. Compute the Jacobian matrix J numerically
        m = length(residual); % Number of data points
        n = length(x);        % Number of parameters
        J = zeros(m, n);      % Initialize Jacobian matrix

        % Compute the Jacobian using grad.m
        for i = 1:m
            % Define scalar function for i-th residual
            scalar_ri = @(xi) phi(xi, t(i)) - y(i); % Residual for the i-th data point
            J(i, :) = grad(scalar_ri, x)';          % Use grad.m to compute the gradient of the i-th residual
        end
        
        % 3.3. Compute search direction
        d = -(J' * J) \ (J' * residual); % Gauss-Newton direction

        % 3.4. Perform line search to determine optimal step length
        fprintf('Gauss-Newton Iteration %d: Starting line search...\n', N_iter);
        F = @(lambda) f(x + lambda * d); % Define F(lambda)
        [lambda, N_eval] = line_search(F, 1, N_eval); % Call line search with initial guess for lambda and update N_eval

        % 3.5. Update parameters
        x = x + lambda * d;

        % 3.6. Check for convergence
        if norm(lambda * d) < tol
            fprintf('Gauss-Newton Iteration %d: Converged with tolerance %.4e.\n', N_iter, tol);
            break; % Exit loop if update step is below the tolerance
        end

        % Print intermediate results if printout flag is set
        if printout
            % Compute the maximum absolute residual
            max_residual = max(abs(residual));
            
            % Compute the gradient norm
            normg = norm(2 * J' * residual);
            
            % Print the iteration details
            fprintf('Gauss-Newton Iteration %d:\n', N_iter);
            fprintf('x = [%.4f, %.4f, %.4f, %.4f]\n', x);
            fprintf('max(abs(r)) = %.4e, norm(grad) = %.4e, lambda = %.4e\n', ...
                    max_residual, normg, lambda);
        end
    end

    % Print final results
    normg = norm(2 * J' * residual); % Final gradient norm
    fprintf('\nFinal Results:\n');
    fprintf('x = [%.4f, %.4f, %.4f, %.4f]\n', x);
    fprintf('Final norm(grad) = %.4e\n', normg);
    fprintf('Total iterations = %d\n', N_iter);           % Number of Gauss-Newton iterations
    fprintf('Total function evaluations = %d\n', N_eval); % Includes evaluations in Gauss-Newton and line search

    elapsed_time = toc; % Stop timing
    fprintf('Total elapsed time: %.4f seconds\n', elapsed_time); % Display elapsed time

    % Plot results if requested
    if plotout
        figure; % Create a new figure
        plot(t, y, 'ro', 'MarkerSize', 6, 'DisplayName', 'Data'); % Plot data points
        hold on;
        plot(t, phi(x, t), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Fitted Curve'); % Plot fitted curve
        legend show; % Add a legend
        title('Gauss-Newton Fit');
        xlabel('t'); % Label x-axis
        ylabel('y'); % Label y-axis
        grid on; % Add grid -> better visualization
    end
end

%% Backtracking line search subroutine with safeguards
function [lambda, N_eval] = line_search(F, lambda0, N_eval)
    % Inputs:
    % F       - Function handle for F(lambda)
    % lambda0 - Initial guess for lambda
    % N_eval  - Function evaluation counter to be updated
    
    % Outputs:
    % lambda  - Step length that minimizes F(lambda)
    % N_eval  - Updated function evaluation counter

    % Parameters
    max_iter = 20;     % Maximum iterations for line search
    alpha = 0.5;       % Backtracking factor
    tol = 1e-6;        % Tolerance for acceptable reduction in F(lambda)
    min_lambda = 1e-8; % Minimum allowed lambda to prevent stall
    lambda = lambda0;  % Initial lambda
    F0 = F(0);         % Initial F(lambda) value
    N_eval = N_eval + 1; % Increment for F(0)

    for iter = 1:max_iter
        F_lambda = F(lambda); % Evaluate F(lambda)
        N_eval = N_eval + 1;  % Increment function evaluation count

        % Safeguard for invalid F(lambda)
        if isnan(F_lambda) || isinf(F_lambda)
            warning('  Line Search Iteration %d: Invalid F(lambda), reducing step size.', iter);
            lambda = alpha * lambda;
            continue;
        end

        % Debugging output: Print only once per iteration
        fprintf('  Line Search Iteration %d: lambda = %.4e, F(lambda) = %.4e\n', iter, lambda, F_lambda);

        % Check for sufficient reduction
        if F_lambda <= F0 * (1 - tol)
            fprintf('  Line Search Iteration %d: Found suitable lambda = %.4e.\n', iter, lambda);
            return; % Accept lambda
        end

        % Reduce lambda
        lambda = alpha * lambda;

        % Terminate if lambda becomes too small
        if lambda < min_lambda
            warning('Line search did not converge. Returning fallback lambda.');
            lambda = min_lambda;
            return;
        end
    end

    % If no acceptable lambda is found, print a warning and fallback
    warning('Line search failed to find a suitable lambda.');
    lambda = min_lambda;
end

%% Line Search Subroutine Test
a_values = [2, -2, 5, -5, 10, -10]; % Test cases for a
initial_guess = 0.1;                % Initial guess for lambda

% Initialize total function evaluations counter
total_N_eval = 0;

for i = 1:length(a_values)
    a = a_values(i); % Assign the current test case
    fprintf('Test %d, a = %d:\n', i, a); % Print current test info

    F = @(lambda) (1 - 10^a * lambda)^2; % Define the test function

    % Initialize test-specific function evaluation counter
    test_N_eval = 0;

    % Perform line search and track function evaluations
    [lambda, test_N_eval] = line_search(F, initial_guess, 0); % Start evaluations from 0 for this test
    total_N_eval = total_N_eval + test_N_eval;                % Add to the total evaluation count

    % Calculate expected lambda
    expected_lambda = 1 / 10^a;
    error = abs(lambda - expected_lambda);
    F_lambda = F(lambda);

    % Print results
    fprintf('Computed lambda = %.5f\n', lambda);
    fprintf('Expected lambda = %.5f\n', expected_lambda);
    fprintf('Error = %.5e\n', error);
    fprintf('F(lambda) = %.5e (should be close to 0)\n', F_lambda);
    fprintf('Function evaluations in this test = %d\n\n', test_N_eval);
end

fprintf('Total function evaluations in all tests = %d\n', total_N_eval);
%% Test 1
[t,y] = data1;
[x,N_eval,N_iter,normg] = gaussnewton(@phi2,t,y,[1;2;3;4],1e-4,1,1);
