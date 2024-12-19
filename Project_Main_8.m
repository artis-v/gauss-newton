%% gaussnewton
function [x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(phi, t, y, x0, tol, printout, plotout)
    tic; % Start timing

    % Inputs:
    % phi          - Function handle for the model function phi(x, t)
    % t            - Vector of independent variable data
    % y            - Vector of dependent variable data (observations)
    % x0           - Initial guess for the parameter vector x
    % tol          - Tolerance for stopping criteria (applied to step size and gradient norm)
    % printout     - Printing intermediate results (1: print, 0: no print)
    % plotout      - Plotting results (1: plot, 0: no plot)
    
    % Outputs:
    % x            - Final estimated parameter vector
    % N_eval       - Total number of function evaluations during the Gauss-Newton process
    % N_iter       - Total number of Gauss-Newton iterations performed
    % max_residual - Maximum absolute residual value at the final parameter estimate (NOT used to check convergence)
    % normg        - Norm of the gradient at the final parameter estimate (used to check convergence)
    % norms        - Norm of the step length at the final parameter estimate (used to check convergence)

    % 1. Initialize variables
    x = x0;         % Start with the initial guess
    N_eval = 0;     % Initialize function evaluation counter
    N_iter = 0;     % Initialize iteration counter
    max_iter = 200; % Set a maximum number of iterations to prevent infinite loops
    epsilon = 1e-6; % Regularization parameter for stability
    
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

            % Increment function evaluations
            N_eval = N_eval + 2*n; % Gradient evaluations using grad.m require two function evaluations (xplus and xminus) for every parameter in x
        end

        % 3.3. Compute search direction
        %d = -(J' * J) \ (J' * residual);                               % Gauss-Newton direction - No regularization term 
        d = -((J' * J) + epsilon * eye(size(J, 2))) \ (J' * residual); % Gauss-Newton direction - With  regularization term for numerical stability

        % 3.4. Perform line search to determine optimal step length
        fprintf('Gauss-Newton Iteration %d: Starting line search...\n', N_iter);
        F = @(lambda) f(x + lambda * d); % Define F(lambda)
        [lambda, temp_N_eval] = line_search(F, 1, 0, n); % Call line search with initial guess for lambda and return line search function evaluations
        N_eval = N_eval + temp_N_eval; % Add line search evaluations to the total count

        % 3.5. Update parameters
        x = x + lambda * d;

        % 3.6. Check for convergence
        % Calculate convergence metrics
        max_residual = max(abs(residual)); % Compute the maximum absolute residual NOTE NOT USED
        norms = norm(lambda * d);          % Compute the Step norm
        normg = norm(2 * J' * residual);   % Compute the Gradient norm

        % Check both criteria
        if norms < tol && normg < tol
            fprintf('Gauss-Newton Iteration %d: Converged with tolerance %.4e (step and gradient).\n', N_iter, tol);
            break; % Exit loop if update step and gradient norm is below the tolerance
        end

        % Print intermediate results if printout flag is set
        if printout
            % Print the iteration details
            fprintf('Gauss-Newton Iteration %d:\n', N_iter);
            fprintf('x = [%.4f, %.4f, %.4f, %.4f]\n', x);
            fprintf('max(abs(r)) = %.4e, norm(grad) = %.4e, norm(step) = %.4e, lambda = %.4e\n', max_residual, normg, norms, lambda);
        end
    end

    % Print final results
    fprintf('\nFinal Results:\n');
    fprintf('x = [%.4f, %.4f, %.4f, %.4f]\n', x);
    fprintf('Final max(abs(r)) = %.4e\n', max_residual);
    fprintf('Final norm(grad) = %.4e\n', normg);
    fprintf('Final norm(step) = %.4e\n', norms);
    fprintf('Total iterations = %d\n', N_iter);
    fprintf('Total function evaluations = %d\n', N_eval);

    elapsed_time = toc; % Stop timing
    fprintf('Total elapsed time: %.4f seconds\n', elapsed_time); % Display elapsed time

    % Plot results if requested
    if plotout
        figure;      % Create a new figure
        plot(t, y, 'ro', 'MarkerSize', 6, 'DisplayName', 'Data'); % Plot data points
        hold on;
        plot(t, phi(x, t), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Fitted Curve'); % Plot fitted curve
        legend show; % Add a legend
        title('Gauss-Newton Fit');
        xlabel('t'); % Label x-axis
        ylabel('y'); % Label y-axis
        grid on;     % Add grid -> better visualization
    end
end

%% Line search subroutine
function [lambda, temp_N_eval] = line_search(F, lambda0, N_eval, n)

    % Inputs:
    % F           - Function handle for F(lambda)
    % lambda0     - Initial guess for lambda
    % N_eval      - Function evaluation counter to be updated
    % n           - Number of parameters in x
    
    % Outputs:
    % lambda      - Step length that minimizes F(lambda)
    % temp_N_eval - Number of function evaluations within the line search

    % Parameters
    alpha = 2;         % Alpha from Alg 3, p.53
    lambda = lambda0;  % Initial lambda
    F0 = F(0);         % Initial F(lambda) value
    gf = grad(F,0);    % F'(0)
    ep = 0.1;          % Epsilon from Alg 3, p.53

    % Increment function evaluations for F(0)
    temp_N_eval = 1;

    % Increment function evaluations for F'(0)
    temp_N_eval = temp_N_eval + 2*n; % Gradient evaluations using grad.m require two function evaluations (xplus and xminus) for every parameter in x

    % Increase lambda while sufficient reduction is not met
    while F(alpha*lambda) < F0 + ep * gf * alpha * lambda
        lambda = alpha * lambda;
        temp_N_eval = temp_N_eval + 1;
        fprintf('  Line Search Iteration: lambda = %.4e, F(lambda) = %.4e\n', lambda, F(lambda));
    end

    % Decrease lambda while F(lambda) is still too large
    while F(lambda) > F0 + ep * gf * lambda
        lambda = lambda / alpha;
        temp_N_eval = temp_N_eval + 1;
        fprintf('  Line Search Backtrack Iteration: lambda = %.4e, F(lambda) = %.4e\n', lambda, F(lambda));
    end

    % Warning to handle invalid or problematic results
    if isnan(F(lambda)) || F(lambda) > F0
        warning('Potential issue with the line search: F(lambda) = %.4e, F(0) = %.4e', F(lambda), F0);
    end

    return;
end

%% Line Search Subroutine Test
a_values = [2, -2, 5, -5, 10, -10]; % Test cases for a
initial_guess = 0.1;                % Initial guess for lambda

for i = 1:length(a_values)
    a = a_values(i); % Assign the current test case
    fprintf('Test %d, a = %d:\n', i, a); % Print current test info

    F = @(lambda) (1 - 10^a * lambda)^2; % Define the test function

    % Perform line search and track function evaluations
    [lambda, test_N_eval] = line_search(F, initial_guess, 0, 0); % Start evaluations from 0 for this test
    total_N_eval = total_N_eval + test_N_eval;                   % Add to the total evaluation count

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
[x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi2,t,y,[1;2;3;4],1e-4,1,1);

%% Diverse initial guesss test
[t,y] = data1;
t1 = [0; 0; 0; 0;];         % Neutral start
t2 = [100; 100; 100; 100];  % Far-from-solution guess
t3 = [-10; 0; -10; 0;];     % Netagive values (x2 and x4 must >=0 for phi2 || x2 must >= for phi1

t_choice = t3;              % Current test

[x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi2,t,y,t_choice,1e-4,1,1);

%% Task - phi1, data1
[t,y] = data1;
[x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi1,t,y,[1;2;3;4],1e-4,1,1);

%% Task - phi1, data2
[t,y] = data2;
[x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi1,t,y,[1;2;3;4],1e-4,1,1);

%% Task - phi2, data1 (Test 1)
[t,y] = data1;
[x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi2,t,y,[1;2;3;4],1e-4,1,1);

%% Task - phi2, data2
[t,y] = data2;
[x,N_eval,N_iter,normg, norms, max_residual] = gaussnewton(@phi1,t,y,[1;2;3;4],1e-4,1,1);

%% Fit phi1 to data1/data2 for multiple starting points
% Format results_phi_data: initial guesses, final values, N_eval, N_iter, max_residual, normg, norms
[t1, y1] = data1;
[t2, y2] = data2;

% Define range of starting points
num_start_points = 2;                            % Number of different starting points
x1_range = linspace(0.1, 20, num_start_points);
x2_range = linspace(0.01, 10, num_start_points); % Reminder: x2>0 required

% Loop over different starting points for data1
results_phi1_data1 = [];
fprintf('Fitting phi1 to data1:\n');
for x1 = x1_range
    for x2 = x2_range
        x0 = [x1; x2]; % Initial guess
        [x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi1, t1, y1, x0, 1e-4, 1, 0);
        results_phi1_data1 = [results_phi1_data1; x0', x', N_eval, N_iter, max_residual, normg, norms];
        fprintf('Initial: [%.2f, %.2f], Final: [%.2f, %.2f], Iterations: %d, max(abs(r)) = %.2e, norm(grad): %.2e, norm(step): %.2e\n', ...
            x1, x2, x(1), x(2), N_iter, max_residual, normg, norms);
    end
end

% Loop over different starting points for data2
results_phi1_data2 = [];
fprintf('\nFitting phi1 to data2:\n');
for x1 = x1_range
    for x2 = x2_range
        x0 = [x1; x2]; % Initial guess
        [x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi1, t2, y2, x0, 1e-4, 1, 0);
        results_phi1_data2 = [results_phi1_data2; x0', x', N_eval, N_iter, max_residual, normg, norms];
        fprintf('Initial: [%.2f, %.2f], Final: [%.2f, %.2f], Iterations: %d, max(abs(r)) = %.2e, norm(grad): %.2e, norm(step): %.2e\n', ...
            x1, x2, x(1), x(2), N_iter, max_residual, normg, norms);
    end
end

%% Fit phi2 to data1/data2 for multiple starting points
% Format results_phi_data: initial guesses, final values, N_eval, N_iter, max_residual, normg, norms
[t1, y1] = data1;
[t2, y2] = data2;

% Define the range of starting points
num_start_points = 3;                            % Number of different starting points
x1_range = linspace(0.1, 20, num_start_points);  
x2_range = linspace(0.01, 10, num_start_points); % Reminder x2>0 required
x3_range = linspace(0.1, 20, num_start_points);  
x4_range = linspace(0.01, 10, num_start_points); % Reminder: x4>0 required

% Loop over different starting points for data1
results_phi2_data1 = [];
fprintf('\nFitting phi2 to data1:\n');
for x1 = x1_range
    for x2 = x2_range
        for x3 = x3_range
            for x4 = x4_range
                x0 = [x1; x2; x3; x4]; % Initial guess
                [x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi2, t1, y1, x0, 1e-4, 0, 0);
                results_phi2_data1 = [results_phi2_data1; x0', x', N_eval, N_iter, max_residual, normg, norms];
                fprintf('Initial: [%.2f, %.2f], Final: [%.2f, %.2f], Iterations: %d, max(abs(r)) = %.2e, norm(grad): %.2e, norm(step): %.2e\n', ...
                    x1, x2, x(1), x(2), N_iter, max_residual, normg, norms);
            end
        end
    end
end

% Fit phi2 to data2 using the best results from phi1 on data2
fprintf('\nFitting phi2 to data2:\n');
results_phi2_data2 = [];

% Find the best x1 and x2 from phi1 on data2 (minimum normg)
[min_value, best_index] = min(results_phi1_data2(:, end));
best_x1_x2 = results_phi1_data2(best_index, 3:4); % Extract best [x1, x2]
best_x1 = best_x1_x2(1);
best_x2 = best_x1_x2(2);

% Iterate over x3 and x4 while keeping x1 and x2 fixed
for x3 = x3_range
    for x4 = x4_range
        x0 = [best_x1; best_x2; x3; x4]; % Use best x1 and x2, vary x3 and x4
        [x, N_eval, N_iter, max_residual, normg, norms] = gaussnewton(@phi2, t2, y2, x0, 1e-4, 0, 0);
        results_phi2_data2 = [results_phi2_data2; x0', x', N_eval, N_iter, max_residual, normg, norms];
        fprintf('Initial: [%.2f, %.2f], Final: [%.2f, %.2f], Iterations: %d, max(abs(r)) = %.2e, norm(grad): %.2e, norm(step): %.2e\n', ...
            x1, x2, x(1), x(2), N_iter, max_residual, normg, norms);
    end
end
