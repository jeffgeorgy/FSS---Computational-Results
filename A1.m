%% Limpa a memoria e carrega os dados
clear
clc
close all
dados = load('A1.txt');
centros = load('a1-ga-cb.txt');
u_ = centros(:,1); % Primeira coordenada dos centros (entrada)
y_ = centros(:,2); % Segunda coordenada dos centros (saída)
u = dados(:,1); % Primeira coordenada dos dados (entrada)
y = dados(:,2); % Segunda coordenada dos dados (saída)
un = (u-min(u))/(max(u)-min(u)); % normalizando
yn = (y-min(y))/(max(y)-min(y)); % normalizando
u__ = (u_-min(u))/(max(u)-min(u)); % normalizando
y__ = (y_-min(y))/(max(y)-min(y)); % normalizando
qt_rotulos = length(u_);
%% Inicializacao do Algoritmo
% Parametros do algoritmo
r = 0.07; % Decaimento Gaussiano
m = 2; % Grau de incerteza
gama = 0.01; % Peso multivariavel
tic % Inicializa a contagem de tempo
% Variaveis do algoritmo
k = 1; % Amostra inicial
R(1) = 1; % Numero de regras inicial
x = [u';y']; % Vetor de dados
n = length(x(:,1)); % Numero de coordenadas
xn = [un';yn']; % Primeiro dado normalizado
xcn{1} = [xn(:,1)]; % Primeiro centro de cluster normalizado
N = length(x(1,:)); % Numero de amostras
Upsilon_c{1}(1) = min([r^2,1]); % Taxa de variacao do primeiro centro
Pc{1}(1) = 1; % Potencial do primeiro centro de cluster
NPC{1}(1) = 0; % Numero de dados do potencial do primeiro centro
psi(1) = 1; % indice desde a ultima criacao de cluster
max_Upsilon(1) = 0; % Valor maximo da taxa de variacao para normalizacao
NF{1}(:,:,1) = eye(n); % Inicializacao do numerador da inversa da matriz de covariancia nebulosa
DF{1}(1) = 1; % Inicializacao do denominador da matriz de covariancia nebulosa
F{1}(:,:,1) = NF{1}(:,:,1)/DF{1}(1);  % Inicializacao da inversa matriz de covariancia nebulosa
S{1}(1) = 1; % Suporte do primeiro cluster
DM{1}(1) = 0; % Distancia media amostral do primeiro cluster
epsilon{1}(1) = r; % Raio do primeiro cluster
S_{1}(1) = 1; % Suporte normalizado do primeiro cluster
s{1}(1) = Pc{k}(1)*exp(-S_{1}(1)); % Fator de sensibilidade do primeiro cluster
deslocamento{1} = xcn{1}(:,1); % Para plotar
mu(1,1) = 1;
muf = mu;
DA{1} = x(:,1);
centros_processo = xcn{1};
%% Realizacao do Algoritmo
for k = 2:N
    k % Mostra o valor de k
    R(k,1) = R(k-1);
    psi(k,1) = psi(k-1);
    xcn{k,1} = xcn{k-1};
    Upsilon_c{k,1} = Upsilon_c{k-1};
    s{k,1} = s{k-1};
    epsilon{k,1} = epsilon{k-1};
    Upsilon(k,1) = norm(xn(:,k) - xn(:,k-1)); % Calcula a norma da primeira diferenca (taxa de variacao)
    if Upsilon(k) > max_Upsilon(k-1) % Se valor for maior que o maximo
        max_Upsilon(k,1) = Upsilon(k); % Atualiza o maximo
    else
        max_Upsilon(k,1) = max_Upsilon(k-1); % Mantem o mesmo valor maximo
    end
    if max_Upsilon(k) > 0
        Upsilon_(k,1) = Upsilon(k)/max_Upsilon(k); % Normaliza a taxa de variacao
    else
        Upsilon_(k,1) = Upsilon(k);
    end
    for i = 1:R(k)  % Para todos os centros de clusters
        dist = sqrt((xn(:,k)-xcn{k}(:,i))'*det(F{k-1}(:,:,i))^(1/n)*inv(F{k-1}(:,:,i))*...
            (xn(:,k)-xcn{k}(:,i))); % Norma induzida
        if isnan(dist) % Para verificacao de erros
            dist
            dist;
        end
        D_(i,k) = exp(dist)-1; % Medicao da distancia adaptativa do dado atual para os centros
    end
    P(k,1) = gama*exp(-4/r^2*Upsilon_(k)^2); % Inicializa o potencial da amostra atual
    eta(k,1) = k-psi(k)-1; % Numero de amostras desde a ultima criacao de cluster
    for j = (psi(k)+1):(k-1) % Para todos os eta(k) dados desde a ultima criacao de cluster
        P(k) = P(k) + 1/(R(k)+eta(k))*(1-gama)*exp(-4/r^2*norm(xn(:,k)...
            -xn(:,j))^2); % Calcula o potencial do dado atual
    end
    for j = 1:R(k) % Para todos os centros de clusters existentes
        P(k,1) = P(k) + 1/(R(k)+eta(k))*(1-gama)*exp(-4/r^2*D_(j,k)^2); % Calcula o potencial do dado atual
        NPC{k}(j,1) = NPC{k-1}(j) + 1; % Atualiza o numero de amostras usadas nos potenciais dos centros
        Pc{k,1}(j,1) = (NPC{k}(j)-1)/NPC{k}(j)*Pc{k-1}(j) + 1/NPC{k}(j)*...
            ((1-gama)*exp(-4/r^2*D_(j,k)^2) + gama*exp(-4/r^2*...
            Upsilon_c{k}(j)^2)); % Atualiza potenciais de centros
    end
    dmin = 1e50; % Inicializa a distancia minima
    for i = 1:R(k) % Para todos os centros de clusters
        if D_(i,k) < dmin  % Se a distancia for menor que a minima
            dmin = D_(i,k); % Atualiza distancia minima
            indc_prox = i; % Indice do cluster mais proximo
        end
    end
    
    if P(k) > (1-s{k}(indc_prox))*Pc{k}(indc_prox) % Condicao para selecao do dado atual
        if dmin < epsilon{k}(indc_prox) % Condicao de proximidade
            rou = P(k)/(P(k)+Pc{k}(indc_prox)); % Peso do dado atual no cruzamento
            xcn{k,1}(:,indc_prox) = rou*xn(:,k) + (1-rou)*xcn{k}...
                (:,indc_prox); % Atualizacao do centro de cluster normalizado
            deslocamento{indc_prox,1} = [deslocamento{indc_prox},xcn{k}(:,indc_prox)];
            centros_processo = [centros_processo,xcn{k}(:,indc_prox)];
            Pc{k}(indc_prox) = rou*P(k)+(1-rou)*Pc{k}(indc_prox); % Atualizacao do potencial do centro
            NPC{k}(indc_prox) = ceil(rou*(R(k)+eta(k))+(1-rou)*NPC{k}...
                (indc_prox)); % Atualizacao do numero NPC
            Upsilon_c{k,1}(indc_prox,1) = rou*Upsilon_(k) + (1-rou)*...
                Upsilon_c{k}(indc_prox); % Atualizacao da taxa de variacao do centro
            dist = sqrt((xn(:,k)-xcn{k}(:,indc_prox))'*det(F{k-1}...
                (:,:,indc_prox))^(1/n)*inv(F{k-1}(:,:,indc_prox))*...
                (xn(:,k)-xcn{k}(:,indc_prox))); % Computa a norma induzida
            D_(indc_prox,k) = exp(dist)-1; % Atualizacao da distancia do dado ao centro
        else % Condicao para criacao de cluster
            R(k,1) = R(k) + 1; % Incrementa o numero de regras
            DA{R(k),1} = [];
            S{k-1}(R(k),1) = 0; % Inicializa o suporte do novo cluster
            DM{k-1}(R(k),1) = 0; % Inicializa a distancia media amostral do novo cluster
            epsilon{k}(R(k),1) = r; % Raio do novo cluster
            S_{k-1}(R(k),1) = 0; % Suporte normalizado do novo cluster
            xcn{k,1}(:,R(k)) = xn(:,k); % Centro normalizado do novo cluster
            deslocamento{R(k),1} = xcn{k}(:,R(k));
            centros_processo = [centros_processo,xcn{k,1}(:,R(k))];
            Pc{k}(R(k),1) = P(k); % Potencial do novo centro
            s{k}(R(k),1) = Pc{k}(R(k))*exp(-S_{k-1}(R(k))); % Fator de sensibilidade do novo cluster
            NPC{k}(R(k),1) = R(k) - 1 + eta(k); % Numero de dados usados para medicao do potencial do novo cluster
            Upsilon_c{k,1}(R(k),1) = Upsilon_(k); % Taxa de variacao do novo centro
            NF{k-1}(:,:,R(k)) = eye(n); % Numerador da inversa da nova matriz de covariancia nebulosa
            DF{k-1}(R(k),1) = 1; % Denominadorda nova matriz de covariancia nebulosa
            F{k-1}(:,:,R(k)) = NF{k-1}(:,:,R(k))/DF{k-1}(R(k)); % Inversa da matriz de covariancia nebulosa
            D_(R(k),k) = 0; % Distancia entre o dado atual e o novo cluster
            psi(k,1) = k; % Atualiza o indice do ultimo centro criado para o atual
            for i = 1:R(k)-1 % Para todos os clusters anteriores
                DM{k-1}(i,1) = (R(k-1))/R(k)*DM{k-1}(i); % Reduz a distancia media amostral
            end
        end
    end
    if R(k) > 1 % Se ha mais de um cluster
        fimcrossover = 0; % Detecta se nao e mais necessario cruzamentos
        while ~fimcrossover % Enquanto for necessario cruzamentos
            houvecrossover = 0; % Detecta se houve cruzamento
            Dc{k,1} = zeros(R(k),R(k)); % Inicializa as distâncias entre centros
            for i = 1:R(k) % Para todos os clusters
                for j = 1:R(k) % Para todos os clusters
                    dist = sqrt((xcn{k}(:,i)-xcn{k}(:,j))'*det(F{k-1}(:,:,i))^(1/n)*inv(F{k-1}(:,:,i))*...
                        (xcn{k}(:,i)-xcn{k}(:,j))); % Computa a norma induzida
                    Dc{k,1}(i,j) = exp(dist)-1; % Computa a distancia entre os centros
                end
            end
            for i = 1:R(k) % Para todos os clusters
                for j = 1:R(k) % Para todos os clusters
                    i_cp = [i,j]; % Vetor de indices para cruzamento
                    if Dc{k,1}(i_cp(1),i_cp(2)) < epsilon{k}(i_cp(1))...
                            && Dc{k,1}(i_cp(2),i_cp(1)) < ...
                            epsilon{k}(i_cp(2)) && i~=j % Se tiver sobreposicao mútua de clusters distintos
                        houvecrossover = 1; % Detecta que houve cruzamento
                        rou = Pc{k}(i_cp(1))/(Pc{k}(i_cp(1))+...
                            Pc{k}(i_cp(2))); % Peso do primeiro cluster no cruzamento
                        xcn{k}(:,R(k)+1) = rou*xcn{k}(:,i_cp(1)) +...
                            (1-rou)*xcn{k}(:,i_cp(2)); % Centro de cluster normalizado resultante
                        deslocamento{R(k)+1,1} = xcn{k}(:,R(k)+1);
                        centros_processo = [centros_processo,xcn{k,1}(:,R(k)+1)];
                        Pc{k}(R(k)+1,1) = rou*Pc{k}(i_cp(1))+(1-rou)...
                            *Pc{k}(i_cp(2)); % Potencial do centro de cluster resultante
                        NPC{k}(R(k)+1,1) = ceil(rou*NPC{k}(i_cp(1))+...
                            (1-rou)*NPC{k}(i_cp(2))); % Numero NPC do centro de cluster resultante
                        Upsilon_c{k}(R(k)+1,1) = rou*Upsilon_c{k}(i_cp(1))...
                            + (1-rou)*Upsilon_c{k}(i_cp(2)); % Taxa de variacao do centro de cluster resultante
                        NF{k-1}(:,:,R(k)+1) = rou*NF{k-1}(:,:,i_cp(1))+(1-rou)*NF{k-1}(:,:,i_cp(2)); % Inversa do numerador da MCN do cluster resultante
                        DF{k-1}(R(k)+1) = rou*DF{k-1}(i_cp(1)) +  (1-rou)*DF{k-1}(i_cp(2)); % Denominador da MCN do cluster resultante
                        F{k-1}(:,:,R(k)+1) = NF{k-1}...
                            (:,:,R(k)+1)/DF{k-1}(R(k)+1); % Inversa da MCN do cluster resultante
                        DA{R(k)+1,1} = [DA{i_cp(1)},DA{i_cp(2)}];
                        S{k-1}(R(k)+1,1) = S{k-1}(i_cp(1)) + S{k-1}(i_cp(2)); % Suporte do cluster resultante
                        S_{k-1}(R(k)+1,1) = S{k-1}(R(k)+1)/max(S{k-1}); % Suporte normalizado do cluster resultante
                        DM{k-1}(R(k)+1,1) = DM{k-1}(i_cp(1)) +...
                            DM{k-1}(i_cp(2)); % Distancia media amostral do cluster resultante
                        epsilon{k}(R(k)+1,1) = DM{k-1}(R(k)+1) +...
                            (r-DM{k-1}(R(k)+1))/...
                            (S{k-1}(R(k)+1)^(1/(2*n))); % Raio do cluster resultante
                        s{k}(R(k)+1,1) = Pc{k}(R(k)+1)*exp(-S_{k-1}...
                            (R(k)+1)); % Fator de sensibilidade do cluster resultante
                        dist = sqrt((xn(:,k)-xcn{k}(:,R(k)+1))'*...
                            det(F{k-1}(:,:,R(k)+1))^(1/n)*inv(F{k-1}...
                            (:,:,R(k)+1))*(xn(:,k)-xcn{k}(:,R(k)+1))); % Distancia de norma induzida
                        D_(R(k)+1,k) = exp(dist)-1; % Distancia do dado atual ao cluster resultante
                        mu(R(k)+1,:) = zeros(size(mu(1,:))); % Inicializa pertinencia do dado atual ao cluster resultante
                        % Remove todas as variaveis dos clusters mesclados
                        xcn{k}(:,max(i_cp)) = [];
                        xcn{k}(:,min(i_cp)) = [];
                        deslocamento(max(i_cp)) = [];
                        deslocamento(min(i_cp)) = [];
                        Pc{k}(max(i_cp)) = [];
                        Pc{k}(min(i_cp)) = [];
                        NPC{k}(max(i_cp)) = [];
                        NPC{k}(min(i_cp)) = [];
                        Upsilon_c{k}(max(i_cp)) = [];
                        Upsilon_c{k}(min(i_cp)) = [];
                        NF{k-1}(:,:,max(i_cp)) = [];
                        NF{k-1}(:,:,min(i_cp)) = [];
                        DF{k-1}(max(i_cp)) = [];
                        DF{k-1}(min(i_cp)) = [];
                        F{k-1}(:,:,max(i_cp)) = [];
                        F{k-1}(:,:,min(i_cp)) = [];
                        DA(max(i_cp)) = [];
                        DA(min(i_cp)) = [];
                        S{k-1}(max(i_cp)) = [];
                        S{k-1}(min(i_cp)) = [];
                        DM{k-1}(max(i_cp)) = [];
                        DM{k-1}(min(i_cp)) = [];
                        epsilon{k}(max(i_cp)) = [];
                        epsilon{k}(min(i_cp)) = [];
                        S_{k-1}(max(i_cp)) = [];
                        S_{k-1}(min(i_cp)) = [];
                        s{k}(max(i_cp)) = [];
                        s{k}(min(i_cp)) = [];
                        D_(max(i_cp),:) = [];
                        D_(min(i_cp),:) = [];
                        mu(max(i_cp),:) = [];
                        mu(min(i_cp),:) = [];
                        R(k) = R(k)-1; % Diminui a quantidade de regras
                        break % Para o laco for interior se houve crossover
                    end
                end
                if houvecrossover % Se houve crossover
                    break % Para o laco for exterior
                end
            end
            if ~houvecrossover % Se nao houve crossover
                fimcrossover = 1; % Determina o fim do processo de crossover
            end
        end
    end
    dmin = 1e50; % Inicializa a distancia minima
    for i = 1:R(k) % Para todos os clusters
        if D_(i,k) < dmin % Se a distancia for menor que a minima
            dmin = D_(i,k); % Atualiza a distancia minima
            indc_prox = i; % Define o cluster mais proximo
        end
    end
    DA{indc_prox,1} = [DA{indc_prox},x(:,k)];
    S{k,1}(indc_prox,1) =  S{k-1}(indc_prox) + 1; % Atualiza o suporte do cluster mais proximo
    DM{k,1}(indc_prox,1) = (S{k}(indc_prox)-1)/S{k}(indc_prox)*DM{k-1}...
        (indc_prox) + 1/S{k}(indc_prox)*D_(indc_prox,k); % Atualiza a distancia media amostral do clusters mais proximo
    for i = 1:R(k) % Para todos os clusters
        if i ~= indc_prox % Se nao for o cluster mais proximo
            S{k}(i,1) = S{k-1}(i); % Mantem o valor do suporte
            DM{k}(i,1) = DM{k-1}(i); % Mantem o valor da distancia media amostral
        end
    end
    epsilon{k,1} = DM{k} + (r-DM{k})./(S{k}.^(1/(2*n))); % Atualiza o raio dos clusters
    for i = 1:R(k) % Para todos os clusters
        S_{k,1}(i,1) = S{k}(i)/max(S{k}); % Atualiza o suporte normalizado
        s{k,1}(i,1) = Pc{k}(i)*exp(-S_{k}(i)); % Atualiza o fator de sensibilidade
    end
    indcn = 0; % Inicializa o indice de cluster com distancia nula
    for i = 1:R(k) % Para todos os clusters
        if D_(i,k) == 0 % Se a distancia for nula
            indcn = i; % Detecta cluster com distancia nula
        end
    end
    if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
        for i = 1:R(k) % Para todos os clusters
            if ~isinf(D_(i,k)) % Se a distancia nao for infinita
                mu(i,k) = 0; % Inicializa pertinencia
                for j = 1:R(k) % Para todos os clusters
                    mu(i,k) = mu(i,k) + (D_(i,k)/D_(j,k))^(2/(m-1)); % Calcula o inverso da pertinencia
                end
                mu(i,k) = 1/mu(i,k); % Calcula a pertinencia
            else % Se a distancia for infinita
                mu(i,k) = 0; % Define pertinencia nula
            end
            muf(i,k) = mu(i,k);
        end
    else % Se houver centro de cluster igual ao dado atual
        mu(indcn,k) = 1; % Define pertinencia unitaria para este cluster
        muf(indcn,k) = mu(indcn,k);
    end
    for i = 1:R(k) % Para todos os clusters
        NF{k,1}(:,:,i) = NF{k-1}(:,:,i) + mu(i,k)^m*(xn(:,k)-xcn{k}(:,i))*(xn(:,k)-xcn{k}(:,i))';
        DF{k,1}(i,1) = DF{k-1}(i) + mu(i,k)^(m); % Atualiza o denominador da MCN
        F{k,1}(:,:,i) = NF{k}(:,:,i)/DF{k}(i); % Atualiza a inversa da MCN
    end
end
toc % Finaliza contagem de tempo
xc{1}(1,:) = xcn{end}(1,:)*(max(x(1,:))-min(x(1,:))) + min(x(1,:));
xc{1}(2,:) = xcn{end}(2,:)*(max(x(2,:))-min(x(2,:))) + min(x(2,:));
for i = 1:n
    centros_processo(i,:) = centros_processo(i,:)*(max(x(i,:))-min(x(i,:))) + min(x(i,:));
end
%% Cores para plotagens com muitas curvas
colors = [
    1.0, 0.0, 0.0;    % Vermelho
    0.0, 1.0, 0.0;    % Verde
    0.0, 0.0, 1.0;    % Azul
    1.0, 1.0, 0.0;    % Amarelo
    1.0, 0.0, 1.0;    % Magenta
    0.0, 1.0, 1.0;    % Ciano
    0.5, 0.5, 0.0;    % Oliva
    0.5, 0.0, 0.5;    % Roxo
    0.0, 0.5, 0.5;    % Verde-azulado
    0.5, 0.5, 0.5;    % Cinza
    1.0, 0.5, 0.0;    % Laranja
    0.0, 0.5, 1.0;    % Azul claro
    0.5, 0.0, 0.0;    % Marrom
    0.0, 0.5, 0.0;    % Verde escuro
    0.0, 0.0, 0.5;    % Azul escuro
    1.0, 0.75, 0.8;   % Rosa claro
    0.5, 0.75, 1.0;   % Azul céu
    0.75, 1.0, 0.75;  % Verde claro
    1.0, 0.5, 0.5;    % Coral
    0.75, 0.75, 0.0;  % Dourado
    0.75, 0.5, 0.25;  % Bronze
    0.25, 0.25, 0.75; % Índigo
    0.5, 0.25, 0.75;  % Lilás
    0.25, 0.75, 0.5;  % Verde menta
    0.75, 0.25, 0.5;  % Carmim
    0.25, 0.25, 0.25; % Preto
    0.25, 0.75, 0.75; % Turquesa
    0.75, 0.25, 0.75; % Fúcsia
    0.5, 0.75, 0.25;  % Pistache
    0.75, 0.75, 0.75; % Cinza claro
    0.75, 0.25, 0.25; % Vermelho terroso
    0.25, 0.5, 0.75;  % Azul oceano
    0.25, 0.75, 0.25; % Verde limão
    0.75, 0.5, 0.5;   % Salmão
    0.25, 0.25, 0.5;  % Azul meia-noite
    0.25, 0.5, 0.25;  % Verde militar
    0.5, 0.25, 0.25;  % Marrom avermelhado
    0.25, 0.75, 1.0;  % Azul piscina
    1.0, 0.5, 0.75;   % Rosa choque
    0.75, 1.0, 0.5;   % Verde maçã
    1.0, 1.0, 0.5;    % Amarelo pastel
    1.0, 0.75, 0.25;  % Dourado claro
    0.5, 0.25, 1.0;   % Roxo vibrante
    0.25, 1.0, 0.5;   % Verde primavera
];

%% Figura 1 (a) - Centros e Protótipos
figure
plot(x(1,:),x(2,:),'c*','LineWidth',1)
hold on
plot(u_,y_,'ko',xc{end}(1,:),xc{end}(2,:),'r*','LineWidth',1)
xm = (-0.1:0.005:1.1)';
xmnn = xm*(max(x(1,:))-min(x(1,:))) + min(x(1,:));
ym = (-0.1:0.005:1.1)';
ymnn = ym*(max(x(2,:))-min(x(2,:))) + min(x(2,:));
for l = 1:R(end)
    l
    for i = 1:length(xm)
        for j = 1:length(ym)
            dist = sqrt(([xm(i);ym(j)]-xcn{end}(:,l))'*det(F{end}(:,:,l))^(1/n)*inv(F{end}(:,:,l))*...
                ([xm(i);ym(j)]-xcn{end}(:,l)));
            Dm(i,j,l) = exp(dist)-1;
        end
    end
    contour(xmnn,ymnn,Dm(:,:,l)',0.08*[1,1],'r','LineWidth',1);
end

legend("Vectors of Data","Real Centers","Achieved Centers","Prototypes",'Interpreter','latex')
xlabel('$x_k^1$','Interpreter','latex')
ylabel('$x_k^2$','Interpreter','latex')
grid on

% %% Figura 1 (b) - Dados Associados
% figure
% for i = 1:R(end)
%     plot(DA{i}(1,:),DA{i}(2,:),'.','Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% plot(xc{end}(1,:),xc{end}(2,:),'k*','LineWidth',1)
% leg{R(end)+1,1} = "Centros de Grupo $\mathbf{c}_k^i$";
% legend(leg,'Interpreter','latex')
% ylim([0 12e5])
% yticks(0:1e5:12e5)
% xlabel('$x_k^1$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% grid on
% clear leg
%% Figura 1 (c) - Deslocamento de centros por processo de atualização
% figure
% plot(x(1,:),x(2,:),'.','Color',[0.8 0.7 1],'LineWidth',1)
% hold on
% plot(u_,y_,'k*','LineWidth',1)
% for i = 1:R(end)
%     plot(deslocamento{i}(1,2:end),deslocamento{i}(2,2:end),'b*',...
%         deslocamento{i}(1,1),deslocamento{i}(2,1),'g*',...
%         xc{end}(1,i),xc{end}(2,i),'r*','LineWidth',1);
% end
% plot(u_,y_,'k*','LineWidth',1)
% for i = 1:R(end)
%     plot(xc{end}(1,i),xc{end}(2,i),'r*','LineWidth',1);
% end
% legend("Dados $\mathbf{x}_k$","Centros de Grupo Reais","Centros $\mathbf{c}_k^i$ Intermedi\'{a}rios"...
%     ,"Centros $\mathbf{c}_k^i$ Iniciais","Centros $\mathbf{c}_k^i$ Finais",'Interpreter','latex')
% xlabel('$x_k^1$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% grid on

figure
plot(x(end-1,:),x(end,:),'c*','LineWidth',1)
hold on
plot(centros_processo(end-1,:),centros_processo(end,:),'b*','LineWidth',1)
plot(u_,y_,'ko','LineWidth',1)
plot(xc{end}(end-1,:),xc{end}(end,:),'r*','LineWidth',1)
legend("Vectors of Data","Centers in the Process",...
    "Real Centers","Final Centers",'Interpreter','latex')
xlabel('$x_k^1$','Interpreter','latex')
ylabel('$x_k^2$','Interpreter','latex')
grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);
% xticks_ = get(gca, 'XTick'); % Obter valores dos ticks no eixo Y
% xticklabels = strrep(strtrim(cellstr(num2str(xticks_.', '%.2f'))), '.', ',');
% set(gca, 'XTickLabel', xticklabels);

%% Figura 1 (d) - Niveis de Contorno
% Gera curva de pertinencias
figure
plot(x(1,:),x(2,:),'c*','LineWidth',1)
hold on
plot(xc{end}(1,:),xc{end}(2,:),'r*','LineWidth',1)
xm = (-0.1:0.005:1.1)';
xmnn = xm*(max(x(1,:))-min(x(1,:))) + min(x(1,:));
ym = (-0.1:0.005:1.1)';
ymnn = ym*(max(x(2,:))-min(x(2,:))) + min(x(2,:));
for i = 1:length(xm)
    i
    for j = 1:length(ym)
        for l = 1:R(end)
            Dm(i,j,l) = exp(sqrt(([xm(i);ym(j)]-xcn{end}(:,l))'*det(F{end}(:,:,l))^(1/n)*inv(F{end}(:,:,l))*...
                ([xm(i);ym(j)]-xcn{end}(:,l))))-1;
        end
        % Calcula a pertinencia do dado k em relação a todos os clusters
        indcn = 0; % Inicializa o indice de cluster com distancia nula
        for l = 1:R(end) % Para todos os clusters
            if Dm(i,j,l) == 0 % Se a distancia for nula
                indcn = l; % Detecta cluster com distancia nula
            end
        end
        if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
            for l = 1:R(end) % Para todos os clusters
                if ~isinf(Dm(i,j,l)) % Se a distancia nao for infinita
                    mum(i,j,l) = 0; % Inicializa pertinencia
                    for ll = 1:R(end) % Para todos os clusters
                        mum(i,j,l) = mum(i,j,l) + (Dm(i,j,l)/Dm(i,j,ll))^(2/(m-1)); % Calcula o inverso da pertinencia
                    end
                    mum(i,j,l) = 1/mum(i,j,l); % Calcula a pertinencia
                else % Se a distancia for infinita
                    mum(i,j,l) = 0; % Define pertinencia nula
                end
            end
        else % Se houver centro de cluster igual ao dado atual
            mum(i,j,indcn) = 1; % Define pertinencia unitaria para este cluster
        end
    end
end
for l = 1:R(end)
    contour(xmnn,ymnn,mum(:,:,l)',0.9*[1,1],'r','LineWidth',1);
    contour(xmnn,ymnn,mum(:,:,l)',0.7*[1,1],'b','LineWidth',1);
    contour(xmnn,ymnn,mum(:,:,l)',0.5*[1,1],'k','LineWidth',1);
end
legend("Vectors of Data","Centers of Clusters",...
    "$\mu$ = 0.9","$\mu$ = 0.7"...
    ,"$\mu$ = 0.5",'Interpreter','latex')
xlabel('$x_k^1$','Interpreter','latex')
ylabel('$x_k^2$','Interpreter','latex')
grid on

% %% Figura 2 (a) - Dados em série temporal
% figure
% subplot(2,1,1)
% plot(t,u/1e5,'k','LineWidth',1)
% xlabel('$k$','Interpreter','latex')
% ylabel('$x_k^1$','Interpreter','latex')
% % axis([0 4130 0 5])
% yticks(0:2.5:10)
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);
% % yticks(0:1:5)
% subplot(2,1,2)
% plot(t,y/1e5,'k','LineWidth',1)
% xlabel('$k$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% % axis([0 4130 -2 2])
% yticks(0:2.5:10)
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

%% Figura 2 (b) - Variação do Número de Grupos
t = (1:N)';
figure
plot(t,R,'b','LineWidth',1)
xlabel('$k$','Interpreter','latex')
ylabel('$R$','Interpreter','latex')
ylim([0 22])
yticks(0:1:22)
grid on

% %% Figura 2 (c) - Potenciais dos dados
% figure
% plot(t(2:end),P(2:end),'k','LineWidth',1)
% xlabel('$k$','Interpreter','latex')
% ylabel('$P_k^{\bar{\mathbf{x}}_k}$','Interpreter','latex')
% % ylim([0 17])
% % yticks(0:1:17)
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

% %% Figura 2 (d) - Potenciais dos centros no proceranksso
% figure
% PotC = zeros(max(R),N);
% for k = 1:N
%     if length(Pc{k}) == max(R)
%         PotC(:,k) = Pc{k};
%     elseif length(Pc{k}) < max(R)
%         PotC(:,k) = [Pc{k};zeros(max(R)-length(Pc{k}),1)];
%     else
%         PotC(:,k) = Pc{k}(1:max(R));
%     end
% end
% for i = 1:max(R)
%     plot(t,PotC(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$P_k^{\bar{\mathbf{c}}_k^i}$','Interpreter','latex')
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

% %% Figura 3 (a) - Variação dos Graus de Pertinência
% 
% figure
% for i = 1:max(R)
%         plot(t,muf(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
% leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$\mu_k^i$','Interpreter','latex')
% ylim([0 1.2]);
% yticks(0:0.1:1.2);
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

% %% Figura 3 (b) - Suportes no processo
% figure
% SC = zeros(max(R),N);
% for k = 1:N
%     if length(S{k}) == max(R)
%         SC(:,k) = S{k};
%     elseif length(S{k}) < max(R)
%         SC(:,k) = [S{k};zeros(max(R)-length(S{k}),1)];
%     else
%         SC(:,k) = S{k}(1:max(R));
%     end
% end
% for i = 1:max(R)
%     plot(t,SC(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$S_k^i$','Interpreter','latex')
% grid on

% %% Figura 3 (c) - Raios dos grupos no processo
% figure
% epsilonC = zeros(max(R),N);
% for k = 1:N
%     if length(epsilon{k}) == max(R)
%         epsilonC(:,k) = epsilon{k};
%     elseif length(Pc{k}) < max(R)
%         epsilonC(:,k) = [epsilon{k};zeros(max(R)-length(epsilon{k}),1)];
%     else
%         epsilonC(:,k) = epsilon{k}(1:max(R));
%     end
% end
% for i = 1:max(R)
%     plot(t,epsilonC(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$\varepsilon_k^i$','Interpreter','latex')
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

%% Figura 3 (d) - Sensibilidades no processo
figure
sC = zeros(max(R),N);
for k = 1:N
    if length(s{k}) == max(R)
        sC(:,k) = s{k};
    elseif length(s{k}) < max(R)
        sC(:,k) = [s{k};zeros(max(R)-length(s{k}),1)];
    else
        sC(:,k) = s{k}(1:max(R));
    end
end
for i = 1:max(R)
    plot(t,sC(i,:),'Color',colors(i,:),'LineWidth',1);
    if i == 1
        hold on
    end
    leg{i,1} = strcat("Cluster $",num2str(i),"$");
end
legend(leg,'Interpreter','latex')
xlabel('$k$','Interpreter','latex')
ylabel('$\sigma_i$','Interpreter','latex')
ylim([0 0.5]);
yticks(0:0.05:0.5);
grid on


%% Computa o índice de qualidade e de proximidade de centros
% Mede a partição final
mu = zeros(R(end),N);
D_ = mu;
for i = 1:R(end)
    dadosg{i,1} = [];
end
for j = 1:N
    for i = 1:R(end)  % Para todos os centros de clusters
        dist = sqrt((xn(:,j)-xcn{end}(:,i))'*det(F{end}(:,:,i))^(1/n)*inv(F{end}(:,:,i))*...
            (xn(:,j)-xcn{end}(:,i))); % Norma induzida
        D_(i,j) = exp(dist)-1; % Medicao da distancia adaptativa do dado atual para os centros
    end
    indcn = 0; % Inicializa o indice de cluster com distancia nula
    for i = 1:R(end) % Para todos os clusters
        if D_(i,j) == 0 % Se a distancia for nula
            indcn = i; % Detecta cluster com distancia nula
        end
    end
    if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
        for i = 1:R(end) % Para todos os clusters
            if ~isinf(D_(i,j)) % Se a distancia nao for infinita
                mu(i,j) = 0; % Inicializa pertinencia
                for jj = 1:R(end) % Para todos os clusters
                    mu(i,j) = mu(i,j) + (D_(i,j)/D_(jj,j))^(2/(m-1)); % Calcula o inverso da pertinencia
                end
                mu(i,j) = 1/mu(i,j); % Calcula a pertinencia
            else % Se a distancia for infinita
                mu(i,j) = 0; % Define pertinencia nula
            end
        end
    else % Se houver centro de cluster igual ao dado atual
        mu(indcn,j) = 1; % Define pertinencia unitaria para este cluster
    end
    indcmaxmu = 0;
    mumax = 0;
    for i = 1:R(end)
        if mu(i,j) > mumax
            mumax = mu(i,j);
            indcmaxmu = i;
        end
    end
    dadosg{indcmaxmu} = [dadosg{indcmaxmu},x(:,j)];
end

% figure
% for i = 1:R(end)
%     plot(dadosg{i}(1,:),dadosg{i}(2,:),'.','Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% plot(xc{end}(1,:),xc{end}(2,:),'k*','LineWidth',1)
% leg{R(end)+1,1} = "Centros de Grupo $\mathbf{c}_k^i$";
% legend(leg,'Interpreter','latex')
% ylim([0 12e5])
% yticks(0:1e5:12e5)
% xlabel('$x_k^1$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% grid on
% clear leg

dmin = 1e100;
clear Dc
Dc = zeros(R(end),R(end)); % Inicializa as ditâncias
for i = 1:R(end)
    for j = 1:R(end) % Calcula as distâncias entre centros medidas a partir de cada um
        Dc(i,j) = norm(xc{end}(:,i)-xc{end}(:,j));
        if (i ~= j) && (Dc(i,j)<dmin) % Encontra a menor das distancias
            i_cp = [i,j];
            dmin = Dc(i,j);
        end
    end
end
num = 0;
for i = 1:R(end)
    for j = 1:N
        num = num + mu(i,j)^2*norm(xc{end}(:,i)-x(:,j));
    end
end
q_c = num/(N*dmin)

u__ = (u_-min(x(1,:)))/(max(x(1,:))-min(x(1,:)));
y__ = (y_-min(x(2,:)))/(max(x(2,:))-min(x(2,:)));
for i = 1:qt_rotulos % Para todos clusters encontrados
    for j = 1:R(end) % Para todos os clusters reais
        d(j,i) = norm(xcn{end}(:,j)-[u__(i);y__(i)]); % Cada coluna tem as distâncias do real para os meus
    end
    dist_f(i,1) = min(d(:,i));
end
indice_erro_centro = mean(dist_f)
