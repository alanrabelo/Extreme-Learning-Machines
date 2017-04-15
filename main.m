
clear all;
close all;

% Gerando base X
X = linspace(-6,6,1250);
Y = sin(2*X);

randIndexes = randperm(length(X));
noise = rand(1, length(Y)) .* 0.3;

Y = Y + noise;

% Separando treino de teste
X_treino = X(randIndexes(1:floor(length(X) * 0.8)));
X_teste = X(randIndexes(floor(length(X) * 0.8) + 1:end)); % Deve ser 20x1
Y_treino = Y(randIndexes(1:floor(length(Y) * 0.8)));
Y_teste = Y(randIndexes(floor(length(Y) * 0.8) + 1:end)); % Deve ser 20x1

MSEArray = [];
numberOfTests = 1;
for testCount = 1:numberOfTests
    for q = 1:100
        p = size(X_treino,2);
        % Vetor de pesos aleatórios para a camada oculta
        W1 = rand(q,1);
        repX = repmat(X_treino, q, 1);
        H = W1 * X_treino; % 20x1
        
        H = 1 ./ (1 + exp(-H));
        % inverse = ((H'*H)\H');
        inverse = pinv(H);
        
        % Vetor de pesos
        
        W2 = Y_treino * inverse; % Deve ser 20x1
        
        
        H2 = W1 * X_teste;
        H2 = 1 ./ (1 + exp(-H2));
        
        Y_final = W2 * H2;
        MSE = sqrt(sum((Y_final - Y_teste).^2));
        MSEArray(testCount, q) = MSE;
    end
end

MediumFinalError = sum(MSEArray)./ numberOfTests;


plot(X_teste,Y_final, 'ro'); hold on;
plot(X_teste,Y_teste, 'bo');