#!/bin/bash

ps -p `cat similtext.pid` -o pid,ppid,cmd,%cpu,%mem,stime,user,time