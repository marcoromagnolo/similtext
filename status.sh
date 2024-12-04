#!/bin/bash

ps -p `cat scrapai-wp.pid` -o pid,ppid,cmd,%cpu,%mem,stime,user,time