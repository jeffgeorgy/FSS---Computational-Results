%% Limpa a memoria e carrega os dados
clear % Limpa a memoria
clc % Limpa a janela de comando
close all
% Carrega os dados da planta
load('Nakanishi3.mat');
% Amostras em regime permanente
u = Nakanishi3(:,[5,6,9]);
y = Nakanishi3(:,12);
for i = 1:length(u(1,:))
    mx = max(u(:,i));
    mn = min(u(:,i));
    if mx == mn
        break
    else
        un(:,i) = (u(:,i)-mn)/(mx-mn);
    end
end
yn = (y-min(y))/(max(y)-min(y)); % normalizando
%% Inicializacao do Algoritmo
% Parametros do algoritmo 0.1086
r = 0.35; % Decaimento Gaussiano
m = 2; % Grau de incerteza
gama = 0.5; % Peso multivariavel
tic % Inicializa a contagem de tempo
% Variaveis do algoritmo
k = 1; % Amostra inicial
R = 1; % Numero de regras inicial
x = [u';y']; % Vetor de dados
n = length(x(:,1)); % Numero de coordenadas
xn = [un';yn']; % Primeiro dado normalizado
xcn = xn(:,1); % Primeiro centro de cluster normalizado
N = length(x(1,:)); % Numero de amostras
Upsilon_c(1) = min([r^2,1]); % Taxa de variacao do primeiro centro
Pc(1) = 1; % Potencial do primeiro centro de cluster
NPC(1) = 0; % Numero de dados do potencial do primeiro centro
psi = 1; % indice desde a ultima criacao de cluster
max_Upsilon = 0; % Valor maximo da taxa de variacao para normalizacao
NF(:,:,1) = eye(n); % Inicializacao do numerador da inversa da matriz de covariancia nebulosa
DF(1) = 1; % Inicializacao do denominador da matriz de covariancia nebulosa
F(:,:,1) = NF(:,:,1)/DF(1);  % Inicializacao da inversa matriz de covariancia nebulosa
S(1) = 1; % Suporte do primeiro cluster
DM(1) = 0; % Distancia media amostral do primeiro cluster
epsilon(1) = r; % Raio do primeiro cluster
S_(1) = 1; % Suporte normalizado do primeiro cluster
s(1) = Pc(1)*exp(-S_(1)); % Fator de sensibilidade do primeiro cluster
%% Parametros ODIK/ERA
Ni = 50; % Instante de inicializacao
nent = length(u(1,:));
nsai = length(y(1,:));
p_pm = 2;
v = [u';y'];
alphaf = 10;
betaf = 10;
lambda = 0.95;
fat_alt = 0.01;
%% Realizacao do algoritmo
for k = 2:N
    k % Mostra o valor de k
    Upsilon = norm(xn(:,k) - xn(:,k-1)); % Calcula a norma da primeira diferenca (taxa de variacao)
    if Upsilon > max_Upsilon % Se valor for maior que o maximo
        max_Upsilon = Upsilon; % Atualiza o maximo
    end
    if max_Upsilon > 0
        Upsilon_ = Upsilon/max_Upsilon; % Normaliza a taxa de variacao
    else
        Upsilon_ = Upsilon;
    end
    for i = 1:R  % Para todos os centros de clusters
        dist = sqrt((xn(:,k)-xcn(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
            (xn(:,k)-xcn(:,i))); % Norma induzida
        D_(i,k) = exp(dist)-1; % Medicao da distancia adaptativa do dado atual para os centros
    end
    P = gama*exp(-4/r^2*Upsilon_^2); % Inicializa o potencial da amostra atual
    eta = k-psi-1; % Numero de amostras desde a ultima criacao de cluster
    for j = (psi+1):(k-1) % Para todos os eta dados desde a ultima criacao de cluster
        P = P + 1/(R+eta)*(1-gama)*exp(-4/r^2*norm(xn(:,k)...
            -xn(:,j))^2); % Calcula o potencial do dado atual
    end
    for j = 1:R % Para todos os centros de clusters existentes
        P = P + 1/(R+eta)*(1-gama)*exp(-4/r^2*D_(j,k)^2); % Calcula o potencial do dado atual
        NPC(j,1) = NPC(j) + 1; % Atualiza o numero de amostras usadas nos potenciais dos centros
        Pc(j,1) = (NPC(j)-1)/NPC(j)*Pc(j) + 1/NPC(j)*...
            ((1-gama)*exp(-4/r^2*D_(j,k)^2) + gama*exp(-4/r^2*...
            Upsilon_c(j)^2)); % Atualiza potenciais de centros
    end
    dmin = 1e50; % Inicializa a distancia minima
    for i = 1:R % Para todos os centros de clusters
        if D_(i,k) < dmin  % Se a distancia for menor que a minima
            dmin = D_(i,k); % Atualiza distancia minima
            indc_prox = i; % Indice do cluster mais proximo
        end
    end
    if P > (1-s(indc_prox))*Pc(indc_prox) % Condicao para selecao do dado atual
        if dmin < epsilon(indc_prox) % Condicao de proximidade
            rou = P/(P+Pc(indc_prox)); % Peso do dado atual no cruzamento
            % if k >= 1900
            %     plot(xn(1,1:k),xn(2,1:k),'c*')
            %     hold on
            %     plot(xn(1,k),xn(2,k),'b*')
            %     plot(xcn(1,:),xcn(2,:),'r*')
            %     plot(xcn(1,indc_prox),xcn(2,indc_prox),'g*')
            %     axis([0 1 0 1])
            %     k
            % end
            xcn(:,indc_prox) = rou*xn(:,k) + (1-rou)*xcn...
                (:,indc_prox); % Atualizacao do centro de cluster normalizado
            Pc(indc_prox) = rou*P+(1-rou)*Pc(indc_prox); % Atualizacao do potencial do centro
            NPC(indc_prox) = ceil(rou*(R+eta)+(1-rou)*NPC...
                (indc_prox)); % Atualizacao do numero NPC
            Upsilon_c(indc_prox,1) = rou*Upsilon_ + (1-rou)*...
                Upsilon_c(indc_prox); % Atualizacao da taxa de variacao do centro
            dist = sqrt((xn(:,k)-xcn(:,indc_prox))'*det(F...
                (:,:,indc_prox))^(1/n)*inv(F(:,:,indc_prox))*...
                (xn(:,k)-xcn(:,indc_prox))); % Computa a norma induzida
            D_(indc_prox,k) = exp(dist)-1; % Atualizacao da distancia do dado ao centro
        else % Condicao para criacao de cluster
            R = R + 1; % Incrementa o numero de regras
            S(R,1) = 0; % Inicializa o suporte do novo cluster
            DM(R,1) = 0; % Inicializa a distancia media amostral do novo cluster
            epsilon(R,1) = r; % Raio do novo cluster
            S_(R,1) = 0; % Suporte normalizado do novo cluster
            xcn(:,R) = xn(:,k); % Centro normalizado do novo cluster
            Pc(R,1) = P; % Potencial do novo centro
            s(R,1) = Pc(R)*exp(-S_(R)); % Fator de sensibilidade do novo cluster
            NPC(R,1) = R - 1 + eta; % Numero de dados usados para medicao do potencial do novo cluster
            Upsilon_c(R,1) = Upsilon_; % Taxa de variacao do novo centro
            NF(:,:,R) = eye(n); % Numerador da inversa da nova matriz de covariancia nebulosa
            DF(R,1) = 1; % Denominadorda nova matriz de covariancia nebulosa
            F(:,:,R) = NF(:,:,R)/DF(R); % Inversa da matriz de covariancia nebulosa
            D_(R,k) = 0; % Distancia entre o dado atual e o novo cluster
            psi = k; % Atualiza o indice do ultimo centro criado para o atual
            for i = 1:R-1 % Para todos os clusters anteriores
                DM(i,1) = (R-1)/R*DM(i); % Reduz a distancia media amostral
            end
            if k > Ni % Inicializa parametros do consequente
                P_cov{R,1} = 0;
                Y_m{R,1} = 0;
                for i = 1:(R-1)
                    P_cov{R} = P_cov{R} + 1/(R-1)*P_cov{i};
                    Y_m{R} = Y_m{R} + 1/(R-1)*Y_m{i};
                end
                Z{R,1} = 0;
                Y_0{R,1} = 0;
                Y_{R,1} = 0;
                Y_1{R,1} = 0;
                Y_2{R,1} = 0;
                Y{R,1} = 0;
                H0{R,1} = 0;
                H1{R,1} = 0;
                R_svd{R,1} = 0;
                S_svd{R,1} = 0;
                Sigma{R,1} = 0;
                Sigma_n{R,1} = 0;
                Rn{R,1} = 0;
                Sn{R,1} = 0;
                nf(R,1) = ceil(mean(nf(1:R-1)));
                A_m{R,k-1} = zeros(nf(R),nf(R));
                B_m{R,k-1} = zeros(nf(R),nent);
                C_m{R,k-1} = zeros(nsai,nf(R));
                D_m{R,k-1} = zeros(nsai,nent);
                Yo{R,1} = 0;
                G{R,1} = 0;
                Po{R,1} = 0;
                Yom{R,1} = 0;
                xz{R,1}(:,k-1) = zeros(nf(R),1);
                yer{R,1}(k-1,:) = zeros(1,nsai);
            end
        end
    end
    if R > 1 % Se ha mais de um cluster
        fimcrossover = 0; % Detecta se nao e mais necessario cruzamentos
        while ~fimcrossover % Enquanto for necessario cruzamentos
            houvecrossover = 0; % Detecta se houve cruzamento
            Dc = zeros(R,R); % Inicializa as distâncias entre centros
            for i = 1:R % Para todos os clusters
                for j = 1:R % Para todos os clusters
                    dist = sqrt((xcn(:,i)-xcn(:,j))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
                        (xcn(:,i)-xcn(:,j))); % Computa a norma induzida
                    Dc(i,j) = exp(dist)-1; % Computa a distancia entre os centros
                end
            end
            for i = 1:R % Para todos os clusters
                for j = 1:R % Para todos os clusters
                    i_cp = [i,j]; % Vetor de indices para cruzamento
                    if Dc(i_cp(1),i_cp(2)) < epsilon(i_cp(1))...
                            && Dc(i_cp(2),i_cp(1)) < ...
                            epsilon(i_cp(2)) && i~=j % Se tiver sobreposicao mútua de clusters distintos
                        houvecrossover = 1; % Detecta que houve cruzamento
                        rou = Pc(i_cp(1))/(Pc(i_cp(1))+...
                            Pc(i_cp(2))); % Peso do primeiro cluster no cruzamento
                        xcn(:,R+1) = rou*xcn(:,i_cp(1)) +...
                            (1-rou)*xcn(:,i_cp(2)); % Centro de cluster normalizado resultante
                        Pc(R+1,1) = rou*Pc(i_cp(1))+(1-rou)...
                            *Pc(i_cp(2)); % Potencial do centro de cluster resultante
                        NPC(R+1,1) = ceil(rou*NPC(i_cp(1))+...
                            (1-rou)*NPC(i_cp(2))); % Numero NPC do centro de cluster resultante
                        Upsilon_c(R+1,1) = rou*Upsilon_c(i_cp(1))...
                            + (1-rou)*Upsilon_c(i_cp(2)); % Taxa de variacao do centro de cluster resultante
                        NF(:,:,R+1) = rou*NF(:,:,i_cp(1))+(1-rou)*NF(:,:,i_cp(2)); % Inversa do numerador da MCN do cluster resultante
                        DF(R+1) = rou*DF(i_cp(1)) + (1-rou)*DF(i_cp(2)); % Denominador da MCN do cluster resultante
                        F(:,:,R+1) = NF(:,:,R+1)/DF(R+1); % Inversa da MCN do cluster resultante
                        S(R+1,1) = S(i_cp(1)) + S(i_cp(2)); % Suporte do cluster resultante
                        S_(R+1,1) = S(R+1)/max(S); % Suporte normalizado do cluster resultante
                        DM(R+1,1) = DM(i_cp(1)) +...
                            DM(i_cp(2)); % Distancia media amostral do cluster resultante
                        epsilon(R+1,1) = DM(R+1) +...
                            (r-DM(R+1))/...
                            (S(R+1)^(1/(2*n))); % Raio do cluster resultante
                        s(R+1,1) = Pc(R+1)*exp(-S_...
                            (R+1)); % Fator de sensibilidade do cluster resultante
                        dist = sqrt((xn(:,k)-xcn(:,R+1))'*...
                            det(F(:,:,R+1))^(1/n)*inv(F...
                            (:,:,R+1))*(xn(:,k)-xcn(:,R+1))); % Distancia de norma induzida
                        D_(R+1,k) = exp(dist)-1; % Distancia do dado atual ao cluster resultante
                        mu(R+1,:) = zeros(size(mu(1,:))); % Inicializa pertinencia do dado atual ao cluster resultante
                        % Remove todas as variaveis dos clusters mesclados
                        xcn(:,max(i_cp)) = [];
                        xcn(:,min(i_cp)) = [];
                        Pc(max(i_cp)) = [];
                        Pc(min(i_cp)) = [];
                        NPC(max(i_cp)) = [];
                        NPC(min(i_cp)) = [];
                        Upsilon_c(max(i_cp)) = [];
                        Upsilon_c(min(i_cp)) = [];
                        NF(:,:,max(i_cp)) = [];
                        NF(:,:,min(i_cp)) = [];
                        DF(max(i_cp)) = [];
                        DF(min(i_cp)) = [];
                        F(:,:,max(i_cp)) = [];
                        F(:,:,min(i_cp)) = [];
                        S(max(i_cp)) = [];
                        S(min(i_cp)) = [];
                        DM(max(i_cp)) = [];
                        DM(min(i_cp)) = [];
                        epsilon(max(i_cp)) = [];
                        epsilon(min(i_cp)) = [];
                        S_(max(i_cp)) = [];
                        S_(min(i_cp)) = [];
                        s(max(i_cp)) = [];
                        s(min(i_cp)) = [];
                        D_(max(i_cp),:) = [];
                        D_(min(i_cp),:) = [];
                        mu(max(i_cp),:) = [];
                        mu(min(i_cp),:) = [];
                        if k > Ni % Parametros consequente
                            nf(R+1,1) = rou*nf(i_cp(1)) + (1-rou)*nf(i_cp(2));
                            A_m{R+1,k-1} = zeros(nf(R+1),nf(R+1));
                            B_m{R+1,k-1} = zeros(nf(R+1),nent);
                            C_m{R+1,k-1} = zeros(nsai,nf(R+1));
                            D_m{R+1,k-1} = zeros(nsai,nent);
                            P_cov{R+1,1} = rou*P_cov{i_cp(1)} + (1-rou)*P_cov{i_cp(2)};
                            Y_m{R+1,1} = rou*Y_m{i_cp(1)} + (1-rou)*Y_m{i_cp(2)};
                            Y_0{R+1,1} = 0;
                            Y_{R+1,1} = 0;
                            Y_1{R+1,1} = 0;
                            Y_2{R+1,1} = 0;
                            Y{R+1,1} = 0;
                            H0{R+1,1} = 0;
                            H1{R+1,1} = 0;
                            R_svd{R+1,1} = 0;
                            S_svd{R+1,1} = 0;
                            Sigma{R+1,1} = 0;
                            Sigma_n{R+1,1} = 0;
                            Rn{R+1,1} = 0;
                            Sn{R+1,1} = 0;
                            Yo{R+1,1} = 0;
                            G{R+1,1} = 0;
                            Po{R+1,1} = 0;
                            Yom{R+1,1} = 0;
                            xz{R+1}(:,k-1) = zeros(nf(R+1),1);
                            if k > Ni+1
                                Z{R+1,1} = rou*Z{i_cp(1)} + (1-rou)*Z{i_cp(2)};
                                yer{R+1,1}(k-1,:) = zeros(1,nsai);
                                Z(max(i_cp)) = [];
                                Z(min(i_cp)) = [];
                                yer(max(i_cp)) = [];
                                yer(min(i_cp)) = [];
                            end
                            P_cov(max(i_cp)) = [];
                            P_cov(min(i_cp)) = [];
                            Y_m(max(i_cp)) = [];
                            Y_m(min(i_cp)) = [];
                            Y_0(max(i_cp)) = [];
                            Y_0(min(i_cp)) = [];
                            Y_(max(i_cp),:) = [];
                            Y_(min(i_cp),:) = [];
                            Y_1(max(i_cp),:) = [];
                            Y_1(min(i_cp),:) = [];
                            Y_2(max(i_cp),:) = [];
                            Y_2(min(i_cp),:) = [];
                            Y(max(i_cp),:) = [];
                            Y(min(i_cp),:) = [];
                            H0(max(i_cp)) = [];
                            H0(min(i_cp)) = [];
                            H1(max(i_cp)) = [];
                            H1(min(i_cp)) = [];
                            R_svd(max(i_cp)) = [];
                            R_svd(min(i_cp)) = [];
                            S_svd(max(i_cp)) = [];
                            S_svd(min(i_cp)) = [];
                            Sigma(max(i_cp)) = [];
                            Sigma(min(i_cp)) = [];
                            Sigma_n(max(i_cp)) = [];
                            Sigma_n(min(i_cp)) = [];
                            Rn(max(i_cp)) = [];
                            Rn(min(i_cp)) = [];
                            Sn(max(i_cp)) = [];
                            Sn(min(i_cp)) = [];
                            A_m(max(i_cp),:) = [];
                            A_m(min(i_cp),:) = [];
                            B_m(max(i_cp),:) = [];
                            B_m(min(i_cp),:) = [];
                            C_m(max(i_cp),:) = [];
                            C_m(min(i_cp),:) = [];
                            D_m(max(i_cp),:) = [];
                            D_m(min(i_cp),:) = [];
                            Yo(max(i_cp),:) = [];
                            Yo(min(i_cp),:) = [];
                            G(max(i_cp),:) = [];
                            G(min(i_cp),:) = [];
                            Po(max(i_cp)) = [];
                            Po(min(i_cp)) = [];
                            Yom(max(i_cp)) = [];
                            Yom(min(i_cp)) = [];
                            xz(max(i_cp)) = [];
                            xz(min(i_cp)) = [];
                            nf(max(i_cp)) = [];
                            nf(min(i_cp)) = [];
                        end
                        R = R-1; % Diminui a quantidade de regras
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
    for i = 1:R % Para todos os clusters
        if D_(i,k) < dmin % Se a distancia for menor que a minima
            dmin = D_(i,k); % Atualiza a distancia minima
            indc_prox = i; % Define o cluster mais proximo
        end
    end
    S(indc_prox,1) =  S(indc_prox) + 1; % Atualiza o suporte do cluster mais proximo
    DM(indc_prox,1) = (S(indc_prox)-1)/S(indc_prox)*DM...
        (indc_prox) + 1/S(indc_prox)*D_(indc_prox,k); % Atualiza a distancia media amostral do clusters mais proximo
    epsilon = DM + (r-DM)./(S.^(1/(2*n))); % Atualiza o raio dos clusters
    for i = 1:R % Para todos os clusters
        S_(i,1) = S(i)/max(S); % Atualiza o suporte normalizado
        s(i,1) = Pc(i)*exp(-S_(i)); % Atualiza o fator de sensibilidade
    end
    indcn = 0; % Inicializa o indice de cluster com distancia nula
    for i = 1:R % Para todos os clusters
        if D_(i,k) == 0 % Se a distancia for nula
            indcn = i; % Detecta cluster com distancia nula
        end
    end
    if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
        for i = 1:R % Para todos os clusters
            if ~isinf(D_(i,k)) % Se a distancia nao for infinita
                mu(i,k) = 0; % Inicializa pertinencia
                for j = 1:R % Para todos os clusters
                    mu(i,k) = mu(i,k) + (D_(i,k)/D_(j,k))^(2/(m-1)); % Calcula o inverso da pertinencia
                end
                mu(i,k) = 1/mu(i,k); % Calcula a pertinencia
            else % Se a distancia for infinita
                mu(i,k) = 0; % Define pertinencia nula
            end
        end
    else % Se houver centro de cluster igual ao dado atual
        mu(indcn,k) = 1; % Define pertinencia unitaria para este cluster
    end
    for i = 1:R % Para todos os clusters
        NF(:,:,i) = NF(:,:,i) + mu(i,k)^m*(xn(:,k)-xcn(:,i))*(xn(:,k)-xcn(:,i))'; % Atualiza a inversa do numerador da MCN
        DF(i,1) = DF(i) + mu(i,k)^(m); % Atualiza o denominador da MCN
        F(:,:,i) = NF(:,:,i)/DF(i); % Atualiza a inversa da MCN
    end
    if k == Ni % Inicializa os parametros de Markov
        % Mede as pertinencias dos Ni primeiros dados em relacao aos
        % clusters formados
        mu(:,1:Ni) = zeros(R,Ni);
        for j = 1:Ni
            indcn = 0; % Inicializa o indice de cluster com distancia nula
            for i = 1:R % Medicao da distancia adaptativa do dado atual para os centros
                dist = sqrt((xn(:,j)-xcn(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
                    (xn(:,j)-xcn(:,i)));
                D_(i,j) = exp(dist)-1;
                if D_(i,j) == 0 % Se a distancia for nula
                    indcn = i; % Detecta cluster com distancia nula
                end
            end
            if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
                for i = 1:R % Para todos os clusters
                    if ~isinf(D_(i,j)) % Se a distancia nao for infinita
                        mu(i,j) = 0; % Inicializa pertinencia
                        for jj = 1:R % Para todos os clusters
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
        end
        y_v = y((p_pm+1):Ni,:)';
        V = u((p_pm+1):Ni,:)';
        tau_a = p_pm;
        for i = 1:p_pm
            V = [V;v(:,(p_pm+1-i):(Ni-i))];
        end
        for i = 1:R
            W{i,1} = diag(mu(i,(p_pm+1):Ni));
            P_cov{i,1} = (V*W{i}*V')^-1;
            Y_m{i,1} = y_v*W{i}*V'*P_cov{i};
            yves{i,1} = Y_m{i}*V;
            % for j = 1:nent
            %     figure
            %     plot(y_v(j,:))
            %     hold on
            %     plot(yves{i}(j,:),'--')
            % end
            Y_0{i,1} = Y_m{i}(1:nsai,1:nent);
            for j = 1:p_pm
                Y_{i,j} = Y_m{i}(1:nsai,(2+(j-1)*(nent+nsai)):(j*(nent+nsai)+1));
                Y_1{i,j} = Y_{i,j}(:,1:nent);
                Y_2{i,j} = -Y_{i,j}(:,(nent+1):(nsai+nent));
            end
            for j = 1:p_pm
                Y{i,j} = Y_1{i,j} - Y_2{i,j}*Y_0{i};
                for jj = 1:(j-1)
                    Y{i,j} = Y{i,j} - Y_2{i,jj}*Y{i,j-jj};
                end
            end
            for j = (p_pm+1):(alphaf+betaf)
                Y{i,j} = 0;
                for jj = 1:p_pm
                    Y{i,j} = Y{i,j} - Y_2{i,jj}*Y{i,j-jj};
                end
            end
            H0{i,1} = cell2mat(Y(i,1:betaf));
            H1{i,1} = cell2mat(Y(i,2:(betaf+1)));
            for j = 2:alphaf
                H0{i,1} = [H0{i};cell2mat(Y(i,j:(betaf+j-1)))];
                H1{i,1} = [H1{i};cell2mat(Y(i,(j+1):(betaf+j)))];
            end
            [R_svd{i,1},Sigma{i,1},S_svd{i,1}] = svd(H0{i});
            rank(H0{i})
            nf(i,1) = rank(H0{i});
            Sigma_n{i,1} = Sigma{i}(1:nf(i),1:nf(i));
            Rn{i,1} = R_svd{i}(:,1:nf(i));
            Sn{i,1} = S_svd{i}(:,1:nf(i));
            Er = [eye(nent);zeros((betaf-1)*nent,nent)];
            Em = [eye(nsai);zeros((alphaf-1)*nsai,nsai)];
            A_m{i,Ni} = Sigma_n{i}^(-1/2)*Rn{i}'*H1{i}*Sn{i}*Sigma_n{i}^(-1/2);
            B_m{i,Ni} = Sigma_n{i}^(1/2)*Sn{i}'*Er;
            C_m{i,Ni} = Em'*Rn{i}*Sigma_n{i}^(1/2);
            D_m{i,Ni} = Y_0{i};
            for kk = 1:p_pm
                Yo{i,kk} = Y_2{i,kk};
                for j = 1:(kk-1)
                    Yo{i,kk} = Yo{i,kk} - Y_2{i,j}*Yo{i,kk-j};
                end
            end
            for kk = (p_pm+1):(alphaf+betaf)
                Yo{i,kk} = 0;
                for j = 1:p_pm
                    Yo{i,kk} = Yo{i,kk} - Y_2{i,j}*Yo{i,kk-j};
                end
            end
            Po{i,1} = C_m{i,Ni};
            Yom{i,1} = Yo{i,1};
            for kk = 1:(alphaf+betaf-1)
                Po{i,1} = [Po{i};C_m{i,Ni}*A_m{i,Ni}^kk];
                Yom{i,1} = [Yom{i};Yo{i,kk+1}];
            end
            G{i,Ni} = (Po{i}'*Po{i})^-1*Po{i}'*Yom{i};
            % xz{i}(:,Ni-tau_a) = zeros(nf(i),1);
            % for j = (Ni-tau_a+1):Ni
            %     xz{i}(:,j) = A_m{i,Ni}*xz{i}(:,j-1)+B_m{i,Ni}*u(j-1,:)';
            % end
            xz{i,1}(:,Ni) = zeros(nf(i),1);
            % yer{i}(Ni,:) = (C_m{i,Ni}*xz{i}(:,Ni)+D_m{i,Ni}*u(Ni,:)')';
        end
        ye(Ni,:) = 0;
        % for i = 1:R
        %     ye(Ni,:) = ye(Ni,:) + mu(i,Ni)*yer{i}(Ni,:);
        % end
        acionamentos = 0;
        k
    end
    if k > Ni
        ppi = u(k,:)';
        for i = 1:p_pm
            ppi = [ppi;v(:,k-i)];
        end
        ye(k,:) = 0;
        for i = 1:R
            Z{i,1} = ppi'*P_cov{i}/(lambda/mu(i,k)+ppi'*P_cov{i}*ppi);
            P_cov{i,1} = lambda^-1*P_cov{i}*(eye(length(P_cov{i}))-ppi*Z{i});
            Y_m{i,1} = Y_m{i} + (y(k,:)'-Y_m{i}*ppi)*Z{i};
            Y_0{i,1} = Y_m{i}(1:nsai,1:nent);
            for j = 1:p_pm
                Y_{i,j} = Y_m{i}(1:nsai,(j*nent+(j-1)*nsai+1):((j+1)*nent+j*nsai));
                Y_1{i,j} = Y_{i,j}(:,1:nent);
                Y_2{i,j} = -Y_{i,j}(:,(nent+1):(nsai+nent));
            end
            for j = 1:p_pm
                Y{i,j} = Y_1{i,j} - Y_2{i,j}*Y_0{i};
                for jj = 1:(j-1)
                    Y{i,j} = Y{i,j} - Y_2{i,jj}*Y{i,j-jj};
                end
            end
            for j = (p_pm+1):(alphaf+betaf)
                Y{i,j} = 0;
                for jj = 1:p_pm
                    Y{i,j} = Y{i,j} - Y_2{i,jj}*Y{i,j-jj};
                end
            end
            H0{i,1} = cell2mat(Y(i,1:betaf));
            H1{i,1} = cell2mat(Y(i,2:(betaf+1)));
            for j = 2:alphaf
                H0{i,1} = [H0{i};cell2mat(Y(i,j:(betaf+j-1)))];
                H1{i,1} = [H1{i};cell2mat(Y(i,(j+1):(betaf+j)))];
            end
            [R_svd{i,1},Sigma{i,1},S_svd{i,1}] = svd(H0{i});
            Sigma_n{i,1} = Sigma{i}(1:nf(i),1:nf(i));
            Rn{i,1} = R_svd{i}(:,1:nf(i));
            Sn{i,1} = S_svd{i}(:,1:nf(i));
            Er = [eye(nent);zeros((betaf-1)*nent,nent)];
            Em = [eye(nsai);zeros((alphaf-1)*nsai,nsai)];
            A_m{i,k} = Sigma_n{i}^(-1/2)*Rn{i}'*H1{i}*Sn{i}*Sigma_n{i}^(-1/2);
            B_m{i,k} = Sigma_n{i}^(1/2)*Sn{i}'*Er;
            C_m{i,k} = Em'*Rn{i}*Sigma_n{i}^(1/2);
            D_m{i,k} = Y_0{i};
            for ii = 1:p_pm
                Yo{i,ii} = Y_2{i,ii};
                for j = 1:(ii-1)
                    Yo{i,ii} = Yo{i,ii} - Y_2{i,j}*Yo{i,ii-j};
                end
            end
            for ii = (p_pm+1):(alphaf+betaf)
                Yo{i,ii} = 0;
                for j = 1:p_pm
                    Yo{i,ii} = Yo{i,ii} - Y_2{i,j}*Yo{i,ii-j};
                end
            end
            Po{i,1} = C_m{i,k};
            Yom{i,1} = Yo{i,1};
            for j = 1:(alphaf+betaf-1)
                Po{i,1} = [Po{i};C_m{i,k}*A_m{i,k}^j];
                Yom{i,1} = [Yom{i};Yo{i,j+1}];
            end
            G{i,k} = (Po{i}'*Po{i})^-1*Po{i}'*Yom{i};
            mud_bru = 1;
            % for j = 1:nf(i)
            %     for jj = 1:nf(i)
            %         if abs(A_m{i,k}(j,jj)-A_m{i,k-1}(j,jj)) > fat_alt*abs(A_m{i,k-1}(j,jj))
            %             mud_bru = 1;
            %         end
            %     end
            % end
            % for j = 1:nf(i)
            %     for jj = 1:nent
            %         if abs(B_m{i,k}(j,jj)-B_m{i,k-1}(j,jj)) > fat_alt*abs(B_m{i,k-1}(j,jj))
            %             mud_bru = 1;
            %         end
            %     end
            % end
            if mud_bru
                acionamentos = acionamentos + 1;
                xz{i,1}(:,k-tau_a) = zeros(nf(i),1);
                yet(k-tau_a,:) = (C_m{i,k}*xz{i}(:,k-tau_a)+D_m{i,k}*u(k-tau_a,:)')';
                for j = (k-tau_a+1):k
                    xz{i,1}(:,j) = A_m{i,k}*xz{i}(:,j-1)+B_m{i,k}*u(j-1,:)'-G{i,k}*(y(j-1,:)'-yet(j-1,:)');
                    yet(j,:) = (C_m{i,k}*xz{i}(:,j)+D_m{i,k}*u(j,:)')';
                end
            else
                xz{i,1}(:,k) = A_m{i,k}*xz{i}(:,k-1)+B_m{i,k}*u(k-1,:)'-G{i,k}*(y(k-1,:)'-ye(k-1,:)');
            end
            yer{i,1}(k,:) = (C_m{i,k}*xz{i}(:,k)+D_m{i,k}*u(k,:)')';
            ye(k,:) = ye(k,:) + mu(i,k)*yer{i}(k,:);
        end
    end
end
tempo = toc
for i = 1:length(x(:,1))
    mx = max(x(i,:));
    mn = min(x(i,:));
    if mx == mn
        break
    else
        xc(i,:) = xcn(i,:)*(mx-mn)+mn;
    end
end

% figure
% plot(u(:,end),y,'.','Color',[0.8 0.7 1],'LineWidth',1)
% hold on
% plot(xc(end-1,:),xc(end,:),'r*','LineWidth',1)
% xm = (-0.1:0.005:1.1)';
% xmnn = xm*(max_x(1)-min_x(1)) + min_x(1);
% ym = (-0.1:0.005:1.1)';
% ymnn = ym*(max_x(2)-min_x(2)) + min_x(2);
% for l = 1:R
%     l
%     for i = 1:length(xm)
%         for j = 1:length(ym)
%             dist = sqrt(([xm(i);ym(j)]-xcn(:,l))'*det(F(:,:,l))^(1/n)*inv(F(:,:,l))*...
%                 ([xm(i);ym(j)]-xcn(:,l)));
%             Dm(i,j,l) = exp(dist)-1;
%         end
%     end
%     contour(xmnn,ymnn,Dm(:,:,l)',0.1*[1,1],'r--','LineWidth',1);
% end
% % axis([0 1 0 1])
% xlabel('Feature 1')
% ylabel('Feature 2')
% 
% % figure
% % plot(qt_reg,'k','LineWidth',1)
% 
% % Gera curva de pertinencias
% figure
% plot(xn(1,:),xn(2,:),'c*',xcn(1,:),xcn(2,:),'r*','LineWidth',1)
% hold on
% xm = (-0.1:0.005:1.1)';
% ym = (-0.1:0.005:1.1)';
% for i = 1:length(xm)
%     i
%     for j = 1:length(ym)
%         for l = 1:R
%             Dm(i,j,l) = exp(sqrt(([xm(i);ym(j)]-xcn(:,l))'*det(F(:,:,l))^(1/n)*inv(F(:,:,l))*...
%                 ([xm(i);ym(j)]-xcn(:,l))))-1;
%         end
%         % Calcula a pertinencia do dado k em relação a todos os clusters
%         indcn = 0; % Inicializa o indice de cluster com distancia nula
%         for l = 1:R % Para todos os clusters
%             if Dm(i,j,l) == 0 % Se a distancia for nula
%                 indcn = l; % Detecta cluster com distancia nula
%             end
%         end
%         if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
%             for l = 1:R % Para todos os clusters
%                 if ~isinf(Dm(i,j,l)) % Se a distancia nao for infinita
%                     mum(i,j,l) = 0; % Inicializa pertinencia
%                     for ll = 1:R % Para todos os clusters
%                         mum(i,j,l) = mum(i,j,l) + (Dm(i,j,l)/Dm(i,j,ll))^(2/(m-1)); % Calcula o inverso da pertinencia
%                     end
%                     mum(i,j,l) = 1/mum(i,j,l); % Calcula a pertinencia
%                 else % Se a distancia for infinita
%                     mum(i,j,l) = 0; % Define pertinencia nula
%                 end
%             end
%         else % Se houver centro de cluster igual ao dado atual
%             mum(i,j,indcn) = 1; % Define pertinencia unitaria para este cluster
%         end
%     end
% end
% for l = 1:R
%     contour(xm,ym,mum(:,:,l)',0.8*[1,1],'r');
%     contour(xm,ym,mum(:,:,l)',0.5*[1,1],'b');
%     contour(xm,ym,mum(:,:,l)',0.2*[1,1],'k');
% end
% legend("Samples", "Clusters centers","\mu = 0.9",...
%     "\mu = 0.7","\mu = 0.5");
% xlabel('Feature 1')
% ylabel('Feature 2')
% t = (1:N)';
% figure
% plot(t,y,'k',t,ye,'g--','LineWidth',1)


% mu = zeros(R,N);
% D_ = mu;
% for j = 1:N
%     for i = 1:R  % Para todos os centros de clusters
%         dist = sqrt((xn(:,j)-xcn(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
%             (xn(:,j)-xcn(:,i))); % Norma induzida
%         D_(i,j) = exp(dist)-1; % Medicao da distancia adaptativa do dado atual para os centros
%     end
%     indcn = 0; % Inicializa o indice de cluster com distancia nula
%     for i = 1:R % Para todos os clusters
%         if D_(i,j) == 0 % Se a distancia for nula
%             indcn = i; % Detecta cluster com distancia nula
%         end
%     end
%     if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
%         for i = 1:R % Para todos os clusters
%             if ~isinf(D_(i,j)) % Se a distancia nao for infinita
%                 mu(i,j) = 0; % Inicializa pertinencia
%                 for jj = 1:R % Para todos os clusters
%                     mu(i,j) = mu(i,j) + (D_(i,j)/D_(jj,j))^(2/(m-1)); % Calcula o inverso da pertinencia
%                 end
%                 mu(i,j) = 1/mu(i,j); % Calcula a pertinencia
%             else % Se a distancia for infinita
%                 mu(i,j) = 0; % Define pertinencia nula
%             end
%         end
%     else % Se houver centro de cluster igual ao dado atual
%         mu(indcn,j) = 1; % Define pertinencia unitaria para este cluster
%     end
% end
% dmin = 1e100;
% Dc = zeros(R,R); % Inicializa as ditâncias
% for i = 1:R
%     for j = 1:R % Calcula as distâncias entre centros medidas a partir de cada um
%         % Dc(i,j) = norm(xcn(:,i)-xcn(:,j));
%         Dc(i,j) = norm(xc(:,i)-xc(:,j));
%         if (i ~= j) && (Dc(i,j)<dmin) % Encontra a menor das distancias
%             i_cp = [i,j];
%             dmin = Dc(i,j);
%         end
%     end
% end
% num = 0;
% for i = 1:R
%     for j = 1:N
%         % num = num + mu(i,j)^2*norm(xcn(:,i)-xn(:,j));
%         num = num + mu(i,j)^2*norm(xc(:,i)-x(:,j));
%     end
% end
% q_c = num/(N*dmin)
RMSE = sqrt(mean((y(Ni+1:end)-ye(Ni+1:end)).^2))
num = 0;
den1 = 0;
den2 = 0;
for i = Ni+1:length(y)
    num = num + (y(i)-mean(y(Ni+1:end)))*(ye(i)-mean(ye(Ni+1:end)));
    den1 = den1 + (y(i)-mean(y(Ni+1:end)))^2;
    den2 = den2 + (ye(i)-mean(ye(Ni+1:end)))^2;
end
CCP = num/(sqrt(den1*den2))

num = 0;
den = 0;
for i = Ni+1:length(y)
    num = num + (y(i)-ye(i))^2;
    den = den + (y(i)-mean(y(Ni+1:end)))^2;
end
CD = 1-num/den

%

% for k = 1:N
%     for i = 1:length(Pc)
%         plotar(k,i) = Pc(i);
%     end
% end
% for i = 1:R
%     plotar(:,i) = plotar(:,i).^2;
% end
% ix = [];
% for i = 1:OTIMOSS1.Q
%     if OTIMOSS1.indvd_ord(i,1) > 0.069 && OTIMOSS1.indvd_ord(i,1) < 0.071
%         ix = [ix;i];
%     end
% end