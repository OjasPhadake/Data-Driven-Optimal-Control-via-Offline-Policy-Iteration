%% Data-driven Off-policy PI for LTVDT Systems
clear; clc; close all;

% --- 1. Problem Setup & Parameters ---
n = 2; % State dimension - x_k belongs to R^n
m = 1; % Input dimension - u_k belongs to R^m
k0 = 0;
N = 120; % number of time steps
epsilon = 0.01; % Convergence threshold 
max_iter = 15;

% Weighting matrices
Q = @(k) (0.04*k + 2) * eye(n);
R = @(k) 5 - 0.02*k;
F = 50 * eye(n);

% System matrices (unknown to the algorithm, used only for data generation)
A = @(k) [1, 0.0025*k; -0.1*cos(0.3*k), 1 + (0.05^1.5*sin(0.5*k)*sqrt(k))]; 
B = @(k) [1; (0.1*k + 2)/(0.1*k + 3)] * 0.05; 

% --- 2. Data Collection Phase ---
% Number of groups (l) must satisfy rank condition
% Required rank = m(m+1)/2 + mn + n(n+1)/2 = 1 + 2 + 3 = 6

l = 100; % l should be at least 6. Anything more than that is bonus data. 
data_x = zeros(n, N+1, l);
data_u = zeros(m, N, l);

for j = 1:l
    x = (2*rand(n, 1) - 1); % Initial state in [-1, 1]
    data_x(:, 1, j) = x;
    
    % FIX: Draw frequencies for the j-th trial ONCE
    s = 1:500;
    sigma = -500 + 1000*rand(1, 500); 

    for k = k0:N-1
        % Exploration noise
        w_kj = 2 * sum(sin(sigma * k)); 
        
        u = w_kj; % Initial gain L_k,0 is [0,0]^T
        data_u(:, k+1, j) = u;
        data_x(:, k+2, j) = A(k)*data_x(:, k+1, j) + B(k)*u;
    end
end

% --- 3. Off-policy Policy Iteration (Algorithm 1) ---
L = zeros(m, n, N, max_iter + 1); % Initial L is zeros
V_history = cell(max_iter, 1); % so this contains 15 different cells
V_norm_diff = zeros(max_iter, 1); % 15*1 double

% Precompute True P_k for comparison (Numerical benchmark)
P_true = zeros(n, n, N+1);
P_true(:,:,N+1) = F;
for k = N-1:-1:k0
    P_next = P_true(:,:,k+2);
    P_true(:,:,k+1) = Q(k) + A(k)'*P_next*A(k) - ...
        A(k)'*P_next*B(k) * inv(R(k) + B(k)'*P_next*B(k)) * B(k)'*P_next*A(k);
end

for i = 1:max_iter
    % Build Phi and Psi matrices
    V_curr = zeros(n, n, N+1);
    V_curr(:,:,N+1) = F;
    
    for k = N-1:-1:k0
        % Pre-allocate matrices for the current time step k
        Gamma_x_tilde = zeros(l, n*(n+1)/2); 
        Delta_k = zeros(l, m*n);
        Gamma_u_tilde = zeros(l, m*(m+1)/2);
        Gamma_Lx = zeros(l, m*(m+1)/2);
        Psi_k_data = zeros(l, 1);
        
        for j = 1:l
            xk = data_x(:, k+1, j);
            xk1 = data_x(:, k+2, j);
            uk0 = data_u(:, k+1, j);
            Lk = L(:,:,k+1,i);
            
            % 1. vecs(V_k) features: x_tilde = [x1^2, 2*x1*x2, x2^2]^T 
            % Note: The paper uses a factor of 2 for cross-terms in vecs(B) 
            % x_tilde = [xk(1)^2; 2*xk(1)*xk(2); xk(2)^2]; 
            x_tilde = [xk(1)^2; xk(1)*xk(2); xk(2)^2];
            
            % 2. Delta_k: 2*[(xk' kron xk')(In kron Lk') + (xk' kron uk0')] 
            term1 = kron(xk', xk') * kron(eye(n), Lk');
            term2 = kron(xk', uk0');
            dk = 2 * (term1 + term2);
            
            % 3. Input features: vecs(uk0) and vecs(Lk*xk) 
            % Since m=1, vecs(u) is just u^2
            u_tilde = uk0^2;
            Lx = Lk * xk;
            Lx_tilde = Lx^2;
            
            % 4. Target Psi: xk'*(Q + L'RL)*xk + xk1'*V_next*xk1 
            % V_curr(:,:,k+2) is the value matrix for time k+1
            target_val = xk'*(Q(k) + Lk'*R(k)*Lk)*xk + xk1'*V_curr(:,:,k+2)*xk1;
            
            % Store in batch matrices
            Gamma_x_tilde(j, :) = x_tilde';
            Delta_k(j, :) = dk;
            Gamma_u_tilde(j, :) = u_tilde';
            Gamma_Lx(j, :) = Lx_tilde';
            Psi_k_data(j) = target_val;
        end
        
        % Solve for Theta_i using the data-driven matrix equation 
        % Phi_i columns: [Gamma_x_tilde, Delta, (Gamma_u - Gamma_Lx)]
        % Since xk1 is already on the RHS in Psi_k_data, we don't include it in Phi
        Phi_k_full = [Gamma_x_tilde, Delta_k, (Gamma_u_tilde - Gamma_Lx)];
        
        % Least squares solution 
        % Theta = (Phi_k_full' * Phi_k_full + 1e-4 * eye(size(Phi_k_full,2))) \ (Phi_k_full' * Psi_k_data);
        Theta = pinv(Phi_k_full) * Psi_k_data;
        
        % --- Extract V_k,i and update Gain for next iteration ---
        % Theta = [vecs(V_k); vec(B'V_k+1 A); vecs(B'V_k+1 B)] 
        V_curr(1,1,k+1) = Theta(1); 
        V_curr(1,2,k+1) = Theta(2)/2; % Divide by 2 due to vecs() definition 
        V_curr(2,1,k+1) = Theta(2)/2; 
        V_curr(2,2,k+1) = Theta(3); % So basically first 4 elements of Theta were V_curr
        % This is because in vecs we flatten the matrix hence it is like
        % this here. The next are again 1 value matrices which are BVB and
        % BVA
        
        BV_A = [Theta(4), Theta(5)];
        BV_B = Theta(6);
        
        % Policy Improvement: L_k,i+1 = (R + B'V_next B)^-1 * (B'V_next A) 
        L(:,:,k+1,i+1) = inv(R(k) + BV_B) * BV_A;
    end
    
    V_history{i} = V_curr;
    
    % Check convergence 
    if i > 1
        diff = 0;
        for k = 1:N
            diff = max(diff, norm(V_history{i}(:,:,k) - V_history{i-1}(:,:,k)));
        end
        V_norm_diff(i) = diff;
        if diff < epsilon
            fprintf('Converged at iteration %d\n', i);
            max_iter_actual = i;
            break;
        end
    end
    max_iter_actual = i;
end

%% --- 4. Plotting Results ---

% Fig 1 & 2: Convergence of V_k,i 
figure;
subplot(1,2,1); hold on;
for i = 1:min(4, max_iter_actual)
    err = arrayfun(@(k) norm(V_history{i}(:,:,k) - P_true(:,:,k)), 1:N);
    plot(1:N, err, '.', 'DisplayName', ['Iteration ' num2str(i)], 'MarkerSize', 12);
end
title('Convergence of V_{k,i} (Iter 1-4)'); xlabel('Time Steps'); ylabel('||V_{k,i} - P_k||');
legend; grid on;

subplot(1,2,2); hold on;
for i = 4:min(8, max_iter_actual)
    err = arrayfun(@(k) norm(V_history{i}(:,:,k) - P_true(:,:,k)), 1:N);
    plot(1:N, err, '.', 'DisplayName', ['Iteration ' num2str(i)], 'MarkerSize', 12);
end
title('Convergence of V_{k,i} (Iter 4-8)'); xlabel('Time Steps'); ylabel('||V_{k,i} - P_k||');
legend; grid on;

% Simulate trajectories 
x_opt = zeros(n, N+1); x_init = zeros(n, N+1);
u_opt = zeros(m, N);
x0 = [-2; 1]; 
x_opt(:,1) = x0; x_init(:,1) = x0;

for k = 1:N
    % Proposed controller (last iteration)
    u_opt(:,k) = -L(:,:,k,max_iter_actual) * x_opt(:,k);
    x_opt(:,k+1) = A(k-1)*x_opt(:,k) + B(k-1)*u_opt(:,k);
    
    % Initial controller (L=0)
    u_init = -[0,0] * x_init(:,k);
    x_init(:,k+1) = A(k-1)*x_init(:,k) + B(k-1)*u_init;
end

% Fig 3: State Trajectories 
figure; hold on;
plot(0:N, x_opt(1,:), 'b-', 'DisplayName', 'x1: proposed', 'LineWidth', 1.5);
plot(0:N, x_opt(2,:), 'r-', 'DisplayName', 'x2: proposed', 'LineWidth', 1.5);
plot(0:N, x_init(1,:), 'b--', 'DisplayName', 'x1: initial','LineWidth', 1.5);
plot(0:N, x_init(2,:), 'r--', 'DisplayName', 'x2: initial', 'LineWidth', 1.5);
title('State Trajectories Comparison'); xlabel('Time Steps'); ylabel('States');
legend; grid on;

% Fig 4: Control Input 
figure;
plot(0:N-1, u_opt, 'b', 'LineWidth', 1);
title('Obtained Approximated Optimal Controller'); xlabel('Time Steps'); ylabel('u');
grid on;
