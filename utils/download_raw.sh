#!/usr/bin/env bash

cd /Users/ingrid/RelatedToSchoolWork/Data/kuaishou_contest/

mkdir train
cd train

wget -c http://cl2018.kuaishou.com/preliminary_contest/train/preliminary_visual_train.zip
unzip preliminary_visual_train.zip
wget -c http://cl2018.kuaishou.com/preliminary_contest/train/train_face.txt
wget -c http://cl2018.kuaishou.com/preliminary_contest/train/train_interaction.txt
wget -c http://cl2018.kuaishou.com/preliminary_contest/train/train_text.txt

cd ./
mkdir test
cd test

wget -c http://cl2018.kuaishou.com/preliminary_contest/test/preliminary_visual_test.zip
unzip  preliminary_visual_test.zip
wget -c http://cl2018.kuaishou.com/preliminary_contest/test/test_face.txt
wget -c http://cl2018.kuaishou.com/preliminary_contest/test/test_interaction.txt
wget -c http://cl2018.kuaishou.com/preliminary_contest/test/test_text.txt