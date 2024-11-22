%%%%% This toolbox provides all the necessary tools to reproduce the results presented in our article.
% It includes scripts for data generation, model implementation, and
% numerical experiments. %%%%%

%% Quant results

%%% First you have to choose the correlation factor for the leadfield G
% between this list [0 0.2 0.4 0.6 0.8 0.99] then you have to chooe the
% iSNR (our results have been obtained with a iSNR of 5, 10 and 20). 
% Finally you get to choose the methods you want to use, 1 for IPFirst and
% 2 for CDLFirst.

clear variables;
close all;

addpath('ltfat/')
addpath('fct_tfnmf/')
addpath('fct_cdl/')
addpath('data/')
addpath('toolbox_signal')
addpath('toolbox_general')

ltfatstart

corr = 0.8;
iSNR = 5;
method = 1;

[optimal_transport_uk, SNR_motifs, SNR_atoms]= SAMPTA_methods(corr,iSNR,method);

%% figures

