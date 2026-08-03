(module ;; columnar_multifluid_rgb_step
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (memory (export "memory") 1)
  (func (export "run") (param $count i32) (param $audio_low i32) (param $audio_high i32) (param $managed_time i32) (param $dt i32) (param $audio_mid i32) (param $column_x i32) (param $column_y i32) (param $ink_blue i32) (param $ink_magenta i32) (param $ink_green i32) (param $ink_cyan i32) (param $entity_y i32) (param $entity_x i32) (param $ink_red i32) (param $ink_yellow i32) (param $entity_b_x i32) (param $entity_c_y i32) (param $entity_b_y i32) (param $entity_c_x i32) (param $entity_velocity_x i32) (param $entity_velocity_y i32) (param $entity_b_velocity_x i32) (param $entity_b_velocity_y i32) (param $entity_c_velocity_x i32) (param $entity_c_velocity_y i32) (param $audio_level i32) (param $displacement_velocity i32) (param $displacement i32) (param $rest_surface i32) (param $out0 i32) (param $out1 i32) (param $out2 i32) (param $out3 i32) (param $out4 i32) (param $out5 i32) (param $out6 i32) (param $out7 i32) (param $out8 i32) (param $out9 i32) (param $out10 i32) (param $out11 i32) (param $out12 i32) (param $out13 i32) (param $out14 i32) (param $out15 i32) (param $out16 i32) (param $out17 i32) (param $out18 i32) (param $out19 i32) (param $out20 i32) (param $out21 i32) (param $out22 i32) (param $out23 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    (local $v12 f64)
    (local $v13 f64)
    (local $v14 f64)
    (local $v15 f64)
    (local $v16 f64)
    (local $v17 f64)
    (local $v18 f64)
    (local $v19 f64)
    (local $v20 f64)
    (local $v21 f64)
    (local $v22 f64)
    (local $v23 f64)
    (local $v24 f64)
    (local $v25 f64)
    (local $v26 f64)
    (local $v27 f64)
    (local $v28 f64)
    (local $v29 f64)
    (local $v30 f64)
    (local $v31 f64)
    (local $v32 f64)
    (local $v33 f64)
    (local $v34 f64)
    (local $v35 f64)
    (local $v36 f64)
    (local $v37 f64)
    (local $v38 f64)
    (local $v39 f64)
    (local $v40 f64)
    (local $v41 f64)
    (local $v42 f64)
    (local $v43 f64)
    (local $v44 f64)
    (local $v45 f64)
    (local $v46 f64)
    (local $v47 f64)
    (local $v48 f64)
    (local $v49 f64)
    (local $v50 f64)
    (local $v51 f64)
    (local $v52 f64)
    (local $v53 f64)
    (local $v54 f64)
    (local $v55 f64)
    (local $v56 f64)
    (local $v57 f64)
    (local $v58 f64)
    (local $v59 f64)
    (local $v60 f64)
    (local $v61 f64)
    (local $v62 f64)
    (local $v63 f64)
    (local $v64 f64)
    (local $v65 f64)
    (local $v66 f64)
    (local $v67 f64)
    (local $v68 f64)
    (local $v69 f64)
    (local $v70 f64)
    (local $v71 f64)
    (local $v72 f64)
    (local $v73 f64)
    (local $v74 f64)
    (local $v75 f64)
    (local $v76 f64)
    (local $v77 f64)
    (local $v78 f64)
    (local $v79 f64)
    (local $v80 f64)
    (local $v81 f64)
    (local $v82 f64)
    (local $v83 f64)
    (local $v84 f64)
    (local $v85 f64)
    (local $v86 f64)
    (local $v87 f64)
    (local $v88 f64)
    (local $v89 f64)
    (local $v90 f64)
    (local $v91 f64)
    (local $v92 f64)
    (local $v93 f64)
    (local $v94 f64)
    (local $v95 f64)
    (local $v96 f64)
    (local $v97 f64)
    (local $v98 f64)
    (local $v99 f64)
    (local $v100 f64)
    (local $v101 f64)
    (local $v102 f64)
    (local $v103 f64)
    (local $v104 f64)
    (local $v105 f64)
    (local $v106 f64)
    (local $v107 f64)
    (local $v108 f64)
    (local $v109 f64)
    (local $v110 f64)
    (local $v111 f64)
    (local $v112 f64)
    (local $v113 f64)
    (local $v114 f64)
    (local $v115 f64)
    (local $v116 f64)
    (local $v117 f64)
    (local $v118 f64)
    (local $v119 f64)
    (local $v120 f64)
    (local $v121 f64)
    (local $v122 f64)
    (local $v123 f64)
    (local $v124 f64)
    (local $v125 f64)
    (local $v126 f64)
    (local $v127 f64)
    (local $v128 f64)
    (local $v129 f64)
    (local $v130 f64)
    (local $v131 f64)
    (local $v132 f64)
    (local $v133 f64)
    (local $v134 f64)
    (local $v135 f64)
    (local $v136 f64)
    (local $v137 f64)
    (local $v138 f64)
    (local $v139 f64)
    (local $v140 f64)
    (local $v141 f64)
    (local $v142 f64)
    (local $v143 f64)
    (local $v144 f64)
    (local $v145 f64)
    (local $v146 f64)
    (local $v147 f64)
    (local $v148 f64)
    (local $v149 f64)
    (local $v150 f64)
    (local $v151 f64)
    (local $v152 f64)
    (local $v153 f64)
    (local $v154 f64)
    (local $v155 f64)
    (local $v156 f64)
    (local $v157 f64)
    (local $v158 f64)
    (local $v159 f64)
    (local $v160 f64)
    (local $v161 f64)
    (local $v162 f64)
    (local $v163 f64)
    (local $v164 f64)
    (local $v165 f64)
    (local $v166 f64)
    (local $v167 f64)
    (local $v168 f64)
    (local $v169 f64)
    (local $v170 f64)
    (local $v171 f64)
    (local $v172 f64)
    (local $v173 f64)
    (local $v174 f64)
    (local $v175 f64)
    (local $v176 f64)
    (local $v177 f64)
    (local $v178 f64)
    (local $v179 f64)
    (local $v180 f64)
    (local $v181 f64)
    (local $v182 f64)
    (local $v183 f64)
    (local $v184 f64)
    (local $v185 f64)
    (local $v186 f64)
    (local $v187 f64)
    (local $v188 f64)
    (local $v189 f64)
    (local $v190 f64)
    (local $v191 f64)
    (local $v192 f64)
    (local $v193 f64)
    (local $v194 f64)
    (local $v195 f64)
    (local $v196 f64)
    (local $v197 f64)
    (local $v198 f64)
    (local $v199 f64)
    (local $v200 f64)
    (local $v201 f64)
    (local $v202 f64)
    (local $v203 f64)
    (local $v204 f64)
    (local $v205 f64)
    (local $v206 f64)
    (local $v207 f64)
    (local $v208 f64)
    (local $v209 f64)
    (local $v210 f64)
    (local $v211 f64)
    (local $v212 f64)
    (local $v213 f64)
    (local $v214 f64)
    (local $v215 f64)
    (local $v216 f64)
    (local $v217 f64)
    (local $v218 f64)
    (local $v219 f64)
    (local $v220 f64)
    (local $v221 f64)
    (local $v222 f64)
    (local $v223 f64)
    (local $v224 f64)
    (local $v225 f64)
    (local $v226 f64)
    (local $v227 f64)
    (local $v228 f64)
    (local $v229 f64)
    (local $v230 f64)
    (local $v231 f64)
    (local $v232 f64)
    (local $v233 f64)
    (local $v234 f64)
    (local $v235 f64)
    (local $v236 f64)
    (local $v237 f64)
    (local $v238 f64)
    (local $v239 f64)
    (local $v240 f64)
    (local $v241 f64)
    (local $v242 f64)
    (local $v243 f64)
    (local $v244 f64)
    (local $v245 f64)
    (local $v246 f64)
    (local $v247 f64)
    (local $v248 f64)
    (local $v249 f64)
    (local $v250 f64)
    (local $v251 f64)
    (local $v252 f64)
    (local $v253 f64)
    (local $v254 f64)
    (local $v255 f64)
    (local $v256 f64)
    (local $v257 f64)
    (local $v258 f64)
    (local $v259 f64)
    (local $v260 f64)
    (local $v261 f64)
    (local $v262 f64)
    (local $v263 f64)
    (local $v264 f64)
    (local $v265 f64)
    (local $v266 f64)
    (local $v267 f64)
    (local $v268 f64)
    (local $v269 f64)
    (local $v270 f64)
    (local $v271 f64)
    (local $v272 f64)
    (local $v273 f64)
    (local $v274 f64)
    (local $v275 f64)
    (local $v276 f64)
    (local $v277 f64)
    (local $v278 f64)
    (local $v279 f64)
    (local $v280 f64)
    (local $v281 f64)
    (local $v282 f64)
    (local $v283 f64)
    (local $v284 f64)
    (local $v285 f64)
    (local $v286 f64)
    (local $v287 f64)
    (local $v288 f64)
    (local $v289 f64)
    (local $v290 f64)
    (local $v291 f64)
    (local $v292 f64)
    (local $v293 f64)
    (local $v294 f64)
    (local $v295 f64)
    (local $v296 f64)
    (local $v297 f64)
    (local $v298 f64)
    (local $v299 f64)
    (local $v300 f64)
    (local $v301 f64)
    (local $v302 f64)
    (local $v303 f64)
    (local $v304 f64)
    (local $v305 f64)
    (local $v306 f64)
    (local $v307 f64)
    (local $v308 f64)
    (local $v309 f64)
    (local $v310 f64)
    (local $v311 f64)
    (local $v312 f64)
    (local $v313 f64)
    (local $v314 f64)
    (local $v315 f64)
    (local $v316 f64)
    (local $v317 f64)
    (local $v318 f64)
    (local $v319 f64)
    (local $v320 f64)
    (local $v321 f64)
    (local $v322 f64)
    (local $v323 f64)
    (local $v324 f64)
    (local $v325 f64)
    (local $v326 f64)
    (local $v327 f64)
    (local $v328 f64)
    (local $v329 f64)
    (local $v330 f64)
    (local $v331 f64)
    (local $v332 f64)
    (local $v333 f64)
    (local $v334 f64)
    (local $v335 f64)
    (local $v336 f64)
    (local $v337 f64)
    (local $v338 f64)
    (local $v339 f64)
    (local $v340 f64)
    (local $v341 f64)
    (local $v342 f64)
    (local $v343 f64)
    (local $v344 f64)
    (local $v345 f64)
    (local $v346 f64)
    (local $v347 f64)
    (local $v348 f64)
    (local $v349 f64)
    (local $v350 f64)
    (local $v351 f64)
    (local $v352 f64)
    (local $v353 f64)
    (local $v354 f64)
    (local $v355 f64)
    (local $v356 f64)
    (local $v357 f64)
    (local $v358 f64)
    (local $v359 f64)
    (local $v360 f64)
    (local $v361 f64)
    (local $v362 f64)
    (local $v363 f64)
    (local $v364 f64)
    (local $v365 f64)
    (local $v366 f64)
    (local $v367 f64)
    (local $v368 f64)
    (local $v369 f64)
    (local $v370 f64)
    (local $v371 f64)
    (local $v372 f64)
    (local $v373 f64)
    (local $v374 f64)
    (local $v375 f64)
    (local $v376 f64)
    (local $v377 f64)
    (local $v378 f64)
    (local $v379 f64)
    (local $v380 f64)
    (local $v381 f64)
    (local $v382 f64)
    (local $v383 f64)
    (local $v384 f64)
    (local $v385 f64)
    (local $v386 f64)
    (local $v387 f64)
    (local $v388 f64)
    (local $v389 f64)
    (local $v390 f64)
    (local $v391 f64)
    (local $v392 f64)
    (local $v393 f64)
    (local $v394 f64)
    (local $v395 f64)
    (local $v396 f64)
    (local $v397 f64)
    (local $v398 f64)
    (local $v399 f64)
    (local $v400 f64)
    (local $v401 f64)
    (local $v402 f64)
    (local $v403 f64)
    (local $v404 f64)
    (local $v405 f64)
    (local $v406 f64)
    (local $v407 f64)
    (local $v408 f64)
    (local $v409 f64)
    (local $v410 f64)
    (local $v411 f64)
    (local $v412 f64)
    (local $v413 f64)
    (local $v414 f64)
    (local $v415 f64)
    (local $v416 f64)
    (local $v417 f64)
    (local $v418 f64)
    (local $v419 f64)
    (local $v420 f64)
    (local $v421 f64)
    (local $v422 f64)
    (local $v423 f64)
    (local $v424 f64)
    (local $v425 f64)
    (local $v426 f64)
    (local $v427 f64)
    (local $v428 f64)
    (local $v429 f64)
    (local $v430 f64)
    (local $v431 f64)
    (local $v432 f64)
    (local $v433 f64)
    (local $v434 f64)
    (local $v435 f64)
    (local $v436 f64)
    (local $v437 f64)
    (local $v438 f64)
    (local $v439 f64)
    (local $v440 f64)
    (local $v441 f64)
    (local $v442 f64)
    (local $v443 f64)
    (local $v444 f64)
    (local $v445 f64)
    (local $v446 f64)
    (local $v447 f64)
    (local $v448 f64)
    (local $v449 f64)
    (local $v450 f64)
    (local $v451 f64)
    (local $v452 f64)
    (local $v453 f64)
    (local $v454 f64)
    (local $v455 f64)
    (local $v456 f64)
    (local $v457 f64)
    (local $v458 f64)
    (local $v459 f64)
    (local $v460 f64)
    (local $v461 f64)
    (local $v462 f64)
    (local $v463 f64)
    (local $v464 f64)
    (local $v465 f64)
    (local $v466 f64)
    (local $v467 f64)
    (local $v468 f64)
    (local $v469 f64)
    (local $v470 f64)
    (local $v471 f64)
    (local $v472 f64)
    (local $v473 f64)
    (local $v474 f64)
    (local $v475 f64)
    (local $v476 f64)
    (local $v477 f64)
    (local $v478 f64)
    (local $v479 f64)
    (local $v480 f64)
    (local $v481 f64)
    (local $v482 f64)
    (local $v483 f64)
    (local $v484 f64)
    (local $v485 f64)
    (local $v486 f64)
    (local $v487 f64)
    (local $v488 f64)
    (local $v489 f64)
    (local $v490 f64)
    (local $v491 f64)
    (local $v492 f64)
    (local $v493 f64)
    (local $v494 f64)
    (local $v495 f64)
    (local $v496 f64)
    (local $v497 f64)
    (local $v498 f64)
    (local $v499 f64)
    (local $v500 f64)
    (local $v501 f64)
    (local $v502 f64)
    (local $v503 f64)
    (local $v504 f64)
    (local $v505 f64)
    (local $v506 f64)
    (local $v507 f64)
    (local $v508 f64)
    (local $v509 f64)
    (local $v510 f64)
    (local $v511 f64)
    (local $v512 f64)
    (local $v513 f64)
    (local $v514 f64)
    (local $v515 f64)
    (local $v516 f64)
    (local $v517 f64)
    (local $v518 f64)
    (local $v519 f64)
    (local $v520 f64)
    (local $v521 f64)
    (local $v522 f64)
    (local $v523 f64)
    (local $v524 f64)
    (local $v525 f64)
    (local $v526 f64)
    (local $v527 f64)
    (local $v528 f64)
    (local $v529 f64)
    (local $v530 f64)
    (local $v531 f64)
    (local $v532 f64)
    (local $v533 f64)
    (local $v534 f64)
    (local $v535 f64)
    (local $v536 f64)
    (local $v537 f64)
    (local $v538 f64)
    (local $v539 f64)
    (local $v540 f64)
    (local $v541 f64)
    (local $v542 f64)
    (local $v543 f64)
    (local $v544 f64)
    (local $v545 f64)
    (local $v546 f64)
    (local $v547 f64)
    (local $v548 f64)
    (local $v549 f64)
    (local $v550 f64)
    (local $v551 f64)
    (local $v552 f64)
    (local $v553 f64)
    (local $v554 f64)
    (local $v555 f64)
    (local $v556 f64)
    (local $v557 f64)
    (local $v558 f64)
    (local $v559 f64)
    (local $v560 f64)
    (local $v561 f64)
    (local $v562 f64)
    (local $v563 f64)
    (local $v564 f64)
    (local $v565 f64)
    (local $v566 f64)
    (local $v567 f64)
    (local $v568 f64)
    (local $v569 f64)
    (local $v570 f64)
    (local $v571 f64)
    (local $v572 f64)
    (local $v573 f64)
    (local $v574 f64)
    (local $v575 f64)
    (local $v576 f64)
    (local $v577 f64)
    (local $v578 f64)
    (local $v579 f64)
    (local $v580 f64)
    (local $v581 f64)
    (local $v582 f64)
    (local $v583 f64)
    (local $v584 f64)
    (local $v585 f64)
    (local $v586 f64)
    (local $v587 f64)
    (local $v588 f64)
    (local $v589 f64)
    (local $v590 f64)
    (local $v591 f64)
    (local $v592 f64)
    (local $v593 f64)
    (local $v594 f64)
    (local $v595 f64)
    (local $v596 f64)
    (local $v597 f64)
    (local $v598 f64)
    (local $v599 f64)
    (local $v600 f64)
    (local $v601 f64)
    (local $v602 f64)
    (local $v603 f64)
    (local $v604 f64)
    (local $v605 f64)
    (local $v606 f64)
    (local $v607 f64)
    (local $v608 f64)
    (local $v609 f64)
    (local $v610 f64)
    (local $v611 f64)
    (local $v612 f64)
    (local $v613 f64)
    (local $v614 f64)
    (local $v615 f64)
    (local $v616 f64)
    (local $v617 f64)
    (local $v618 f64)
    (local $v619 f64)
    (local $v620 f64)
    (local $v621 f64)
    (local $v622 f64)
    (local $v623 f64)
    (local $v624 f64)
    (local $v625 f64)
    (local $v626 f64)
    (local $v627 f64)
    (local $v628 f64)
    (local $v629 f64)
    (local $v630 f64)
    (local $v631 f64)
    (local $v632 f64)
    (local $v633 f64)
    (local $v634 f64)
    (local $v635 f64)
    (local $v636 f64)
    (local $v637 f64)
    (local $v638 f64)
    (local $v639 f64)
    (local $v640 f64)
    (local $v641 f64)
    (local $v642 f64)
    (local $v643 f64)
    (local $v644 f64)
    i32.const 0
    local.set $i
    (block $sum_done_0
      (loop $sum_body_0
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_0
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v6
      local.get $v11
      f64.sub
      local.set $v42
      local.get $v5
      local.get $v12
      f64.sub
      local.set $v43
      local.get $v43
      local.get $v43
      f64.mul
      local.set $v112
      local.get $v42
      local.get $v42
      f64.mul
      local.set $v113
      local.get $v112
      local.get $v113
      f64.add
      local.set $v179
      local.get $v179
      f64.neg
      local.set $v213
      f64.const 0.14
      local.set $v214
      local.get $v179
      local.get $v214
      f64.add
      local.set $v215
      f64.const 4.805000000000001
      local.set $v239
      local.get $v213
      local.get $v239
      f64.div
      local.set $v240
      ;; exp via baked lookup table (see the .wasm)
      local.get $v240
      local.set $v274
      local.get $v274
      local.get $v215
      f64.div
      local.set $v289
      local.get $v296
      local.get $v289
      f64.add
      local.set $v296
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_0
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_1
      (loop $sum_body_1
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_1
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v6
      local.get $v17
      f64.sub
      local.set $v59
      local.get $v5
      local.get $v15
      f64.sub
      local.set $v60
      local.get $v60
      local.get $v60
      f64.mul
      local.set $v133
      local.get $v59
      local.get $v59
      f64.mul
      local.set $v134
      local.get $v133
      local.get $v134
      f64.add
      local.set $v186
      f64.const 0.14
      local.set $v214
      local.get $v186
      f64.neg
      local.set $v221
      local.get $v186
      local.get $v214
      f64.add
      local.set $v222
      f64.const 4.805000000000001
      local.set $v249
      local.get $v221
      local.get $v249
      f64.div
      local.set $v250
      ;; exp via baked lookup table (see the .wasm)
      local.get $v250
      local.set $v279
      local.get $v279
      local.get $v222
      f64.div
      local.set $v287
      local.get $v297
      local.get $v287
      f64.add
      local.set $v297
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_1
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_2
      (loop $sum_body_2
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_2
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v5
      local.get $v18
      f64.sub
      local.set $v70
      local.get $v6
      local.get $v16
      f64.sub
      local.set $v71
      local.get $v70
      local.get $v70
      f64.mul
      local.set $v148
      local.get $v71
      local.get $v71
      f64.mul
      local.set $v149
      local.get $v148
      local.get $v149
      f64.add
      local.set $v193
      f64.const 0.14
      local.set $v214
      local.get $v193
      f64.neg
      local.set $v227
      local.get $v193
      local.get $v214
      f64.add
      local.set $v229
      f64.const 4.805000000000001
      local.set $v258
      local.get $v227
      local.get $v258
      f64.div
      local.set $v259
      ;; exp via baked lookup table (see the .wasm)
      local.get $v259
      local.set $v282
      local.get $v282
      local.get $v229
      f64.div
      local.set $v291
      local.get $v298
      local.get $v291
      f64.add
      local.set $v298
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_2
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_3
      (loop $sum_body_3
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_3
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v0
      local.get $v1
      f64.sub
      local.set $v29
      f64.const 2.0
      local.set $v31
      local.get $v4
      local.get $v31
      f64.mul
      local.set $v32
      f64.const 0.61
      local.set $v33
      local.get $v5
      local.get $v33
      f64.mul
      local.set $v34
      f64.const 0.83
      local.set $v35
      local.get $v6
      local.get $v35
      f64.mul
      local.set $v36
      f64.const 0.37
      local.set $v37
      local.get $v5
      local.get $v37
      f64.mul
      local.set $v38
      f64.const 0.29
      local.set $v39
      local.get $v7
      local.get $v8
      f64.add
      local.set $v40
      local.get $v6
      local.get $v11
      f64.sub
      local.set $v42
      local.get $v5
      local.get $v12
      f64.sub
      local.set $v43
      local.get $v13
      local.get $v14
      f64.add
      local.set $v44
      local.get $v6
      local.get $v39
      f64.mul
      local.set $v45
      f64.const 0.18
      local.set $v82
      local.get $v32
      local.get $v0
      f64.sub
      local.set $v108
      local.get $v29
      local.get $v29
      f64.mul
      local.set $v109
      local.get $v34
      local.get $v36
      f64.add
      local.set $v110
      local.get $v38
      local.get $v45
      f64.sub
      local.set $v111
      local.get $v43
      local.get $v43
      f64.mul
      local.set $v112
      local.get $v42
      local.get $v42
      f64.mul
      local.set $v113
      f64.const 1.35
      local.set $v114
      local.get $v40
      local.get $v114
      f64.mul
      local.set $v115
      f64.const 0.3
      local.set $v116
      local.get $v44
      local.get $v116
      f64.mul
      local.set $v117
      f64.const 0.08
      local.set $v126
      local.get $v108
      local.get $v1
      f64.sub
      local.set $v177
      ;; sin via baked lookup table (see the .wasm)
      local.get $v111
      local.set $v178
      local.get $v112
      local.get $v113
      f64.add
      local.set $v179
      local.get $v115
      local.get $v117
      f64.sub
      local.set $v180
      local.get $v177
      local.get $v177
      f64.mul
      local.set $v202
      f64.const 0.72
      local.set $v210
      local.get $v178
      local.get $v210
      f64.mul
      local.set $v211
      local.get $v179
      local.get $v126
      f64.add
      local.set $v212
      local.get $v179
      f64.neg
      local.set $v213
      f64.const 0.14
      local.set $v214
      local.get $v179
      local.get $v214
      f64.add
      local.set $v215
      local.get $v109
      local.get $v202
      f64.add
      local.set $v236
      local.get $v110
      local.get $v211
      f64.add
      local.set $v237
      local.get $v212
      f64.sqrt
      local.set $v238
      f64.const 4.805000000000001
      local.set $v239
      local.get $v213
      local.get $v239
      f64.div
      local.set $v240
      f64.const 1e-05
      local.set $v271
      local.get $v236
      local.get $v271
      f64.add
      local.set $v272
      ;; cos via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v273
      ;; exp via baked lookup table (see the .wasm)
      local.get $v240
      local.set $v274
      ;; sin via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v275
      local.get $v43
      local.get $v238
      f64.div
      local.set $v276
      local.get $v274
      local.get $v215
      f64.div
      local.set $v289
      local.get $v272
      f64.sqrt
      local.set $v290
      local.get $v29
      local.get $v290
      f64.div
      local.set $v293
      local.get $v177
      local.get $v290
      f64.div
      local.set $v295
      local.get $v273
      local.get $v293
      f64.mul
      local.set $v300
      local.get $v275
      local.get $v295
      f64.mul
      local.set $v301
      local.get $v300
      local.get $v301
      f64.add
      local.set $v308
      local.get $v308
      local.get $v82
      f64.mul
      local.set $v310
      local.get $v180
      local.get $v310
      f64.add
      local.set $v315
      local.get $v289
      local.get $v315
      f64.mul
      local.set $v316
      local.get $v276
      local.get $v316
      f64.mul
      local.set $v318
      local.get $v325
      local.get $v318
      f64.add
      local.set $v325
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_3
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_4
      (loop $sum_body_4
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_4
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v0
      local.get $v1
      f64.sub
      local.set $v29
      f64.const 2.0
      local.set $v31
      local.get $v4
      local.get $v31
      f64.mul
      local.set $v32
      f64.const 0.61
      local.set $v33
      local.get $v5
      local.get $v33
      f64.mul
      local.set $v34
      f64.const 0.83
      local.set $v35
      local.get $v6
      local.get $v35
      f64.mul
      local.set $v36
      f64.const 0.37
      local.set $v37
      local.get $v5
      local.get $v37
      f64.mul
      local.set $v38
      f64.const 0.29
      local.set $v39
      local.get $v7
      local.get $v8
      f64.add
      local.set $v40
      local.get $v6
      local.get $v11
      f64.sub
      local.set $v42
      local.get $v5
      local.get $v12
      f64.sub
      local.set $v43
      local.get $v13
      local.get $v14
      f64.add
      local.set $v44
      local.get $v6
      local.get $v39
      f64.mul
      local.set $v45
      f64.const 0.18
      local.set $v82
      local.get $v32
      local.get $v0
      f64.sub
      local.set $v108
      local.get $v29
      local.get $v29
      f64.mul
      local.set $v109
      local.get $v34
      local.get $v36
      f64.add
      local.set $v110
      local.get $v38
      local.get $v45
      f64.sub
      local.set $v111
      local.get $v43
      local.get $v43
      f64.mul
      local.set $v112
      local.get $v42
      local.get $v42
      f64.mul
      local.set $v113
      f64.const 1.35
      local.set $v114
      local.get $v40
      local.get $v114
      f64.mul
      local.set $v115
      f64.const 0.3
      local.set $v116
      local.get $v44
      local.get $v116
      f64.mul
      local.set $v117
      f64.const 0.08
      local.set $v126
      local.get $v108
      local.get $v1
      f64.sub
      local.set $v177
      ;; sin via baked lookup table (see the .wasm)
      local.get $v111
      local.set $v178
      local.get $v112
      local.get $v113
      f64.add
      local.set $v179
      local.get $v115
      local.get $v117
      f64.sub
      local.set $v180
      local.get $v177
      local.get $v177
      f64.mul
      local.set $v202
      f64.const 0.72
      local.set $v210
      local.get $v178
      local.get $v210
      f64.mul
      local.set $v211
      local.get $v179
      local.get $v126
      f64.add
      local.set $v212
      local.get $v179
      f64.neg
      local.set $v213
      f64.const 0.14
      local.set $v214
      local.get $v179
      local.get $v214
      f64.add
      local.set $v215
      local.get $v109
      local.get $v202
      f64.add
      local.set $v236
      local.get $v110
      local.get $v211
      f64.add
      local.set $v237
      local.get $v212
      f64.sqrt
      local.set $v238
      f64.const 4.805000000000001
      local.set $v239
      local.get $v213
      local.get $v239
      f64.div
      local.set $v240
      f64.const 1e-05
      local.set $v271
      local.get $v236
      local.get $v271
      f64.add
      local.set $v272
      ;; cos via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v273
      ;; exp via baked lookup table (see the .wasm)
      local.get $v240
      local.set $v274
      ;; sin via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v275
      local.get $v42
      local.get $v238
      f64.div
      local.set $v277
      local.get $v274
      local.get $v215
      f64.div
      local.set $v289
      local.get $v272
      f64.sqrt
      local.set $v290
      local.get $v29
      local.get $v290
      f64.div
      local.set $v293
      local.get $v177
      local.get $v290
      f64.div
      local.set $v295
      local.get $v273
      local.get $v293
      f64.mul
      local.set $v300
      local.get $v275
      local.get $v295
      f64.mul
      local.set $v301
      local.get $v300
      local.get $v301
      f64.add
      local.set $v308
      local.get $v308
      local.get $v82
      f64.mul
      local.set $v310
      local.get $v180
      local.get $v310
      f64.add
      local.set $v315
      local.get $v289
      local.get $v315
      f64.mul
      local.set $v316
      local.get $v277
      local.get $v316
      f64.mul
      local.set $v317
      local.get $v326
      local.get $v317
      f64.add
      local.set $v326
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_4
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_5
      (loop $sum_body_5
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_5
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v0
      local.get $v1
      f64.sub
      local.set $v29
      f64.const 2.0
      local.set $v31
      local.get $v4
      local.get $v31
      f64.mul
      local.set $v32
      f64.const 0.61
      local.set $v33
      local.get $v5
      local.get $v33
      f64.mul
      local.set $v34
      f64.const 0.83
      local.set $v35
      local.get $v6
      local.get $v35
      f64.mul
      local.set $v36
      f64.const 0.37
      local.set $v37
      local.get $v5
      local.get $v37
      f64.mul
      local.set $v38
      f64.const 0.29
      local.set $v39
      local.get $v9
      local.get $v10
      f64.add
      local.set $v41
      local.get $v13
      local.get $v14
      f64.add
      local.set $v44
      local.get $v6
      local.get $v39
      f64.mul
      local.set $v45
      local.get $v6
      local.get $v17
      f64.sub
      local.set $v59
      local.get $v5
      local.get $v15
      f64.sub
      local.set $v60
      f64.const 0.18
      local.set $v82
      local.get $v32
      local.get $v0
      f64.sub
      local.set $v108
      local.get $v29
      local.get $v29
      f64.mul
      local.set $v109
      local.get $v34
      local.get $v36
      f64.add
      local.set $v110
      local.get $v38
      local.get $v45
      f64.sub
      local.set $v111
      f64.const 1.35
      local.set $v114
      f64.const 0.3
      local.set $v116
      f64.const 0.08
      local.set $v126
      local.get $v60
      local.get $v60
      f64.mul
      local.set $v133
      local.get $v59
      local.get $v59
      f64.mul
      local.set $v134
      local.get $v44
      local.get $v114
      f64.mul
      local.set $v135
      local.get $v41
      local.get $v116
      f64.mul
      local.set $v136
      local.get $v108
      local.get $v1
      f64.sub
      local.set $v177
      ;; sin via baked lookup table (see the .wasm)
      local.get $v111
      local.set $v178
      local.get $v133
      local.get $v134
      f64.add
      local.set $v186
      local.get $v135
      local.get $v136
      f64.sub
      local.set $v187
      local.get $v177
      local.get $v177
      f64.mul
      local.set $v202
      f64.const 0.72
      local.set $v210
      local.get $v178
      local.get $v210
      f64.mul
      local.set $v211
      f64.const 0.14
      local.set $v214
      local.get $v186
      local.get $v126
      f64.add
      local.set $v220
      local.get $v186
      f64.neg
      local.set $v221
      local.get $v186
      local.get $v214
      f64.add
      local.set $v222
      local.get $v109
      local.get $v202
      f64.add
      local.set $v236
      local.get $v110
      local.get $v211
      f64.add
      local.set $v237
      local.get $v220
      f64.sqrt
      local.set $v248
      f64.const 4.805000000000001
      local.set $v249
      local.get $v221
      local.get $v249
      f64.div
      local.set $v250
      f64.const 1e-05
      local.set $v271
      local.get $v236
      local.get $v271
      f64.add
      local.set $v272
      ;; cos via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v273
      ;; sin via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v275
      local.get $v60
      local.get $v248
      f64.div
      local.set $v278
      ;; exp via baked lookup table (see the .wasm)
      local.get $v250
      local.set $v279
      local.get $v279
      local.get $v222
      f64.div
      local.set $v287
      local.get $v272
      f64.sqrt
      local.set $v290
      local.get $v29
      local.get $v290
      f64.div
      local.set $v293
      local.get $v177
      local.get $v290
      f64.div
      local.set $v295
      local.get $v275
      local.get $v293
      f64.mul
      local.set $v304
      local.get $v273
      local.get $v295
      f64.mul
      local.set $v305
      local.get $v304
      local.get $v305
      f64.sub
      local.set $v309
      local.get $v309
      local.get $v82
      f64.mul
      local.set $v311
      local.get $v187
      local.get $v311
      f64.add
      local.set $v312
      local.get $v287
      local.get $v312
      f64.mul
      local.set $v313
      local.get $v278
      local.get $v313
      f64.mul
      local.set $v314
      local.get $v327
      local.get $v314
      f64.add
      local.set $v327
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_5
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_6
      (loop $sum_body_6
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_6
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v0
      local.get $v1
      f64.sub
      local.set $v29
      f64.const 2.0
      local.set $v31
      local.get $v4
      local.get $v31
      f64.mul
      local.set $v32
      f64.const 0.61
      local.set $v33
      local.get $v5
      local.get $v33
      f64.mul
      local.set $v34
      f64.const 0.83
      local.set $v35
      local.get $v6
      local.get $v35
      f64.mul
      local.set $v36
      f64.const 0.37
      local.set $v37
      local.get $v5
      local.get $v37
      f64.mul
      local.set $v38
      f64.const 0.29
      local.set $v39
      local.get $v9
      local.get $v10
      f64.add
      local.set $v41
      local.get $v13
      local.get $v14
      f64.add
      local.set $v44
      local.get $v6
      local.get $v39
      f64.mul
      local.set $v45
      local.get $v6
      local.get $v17
      f64.sub
      local.set $v59
      local.get $v5
      local.get $v15
      f64.sub
      local.set $v60
      f64.const 0.18
      local.set $v82
      local.get $v32
      local.get $v0
      f64.sub
      local.set $v108
      local.get $v29
      local.get $v29
      f64.mul
      local.set $v109
      local.get $v34
      local.get $v36
      f64.add
      local.set $v110
      local.get $v38
      local.get $v45
      f64.sub
      local.set $v111
      f64.const 1.35
      local.set $v114
      f64.const 0.3
      local.set $v116
      f64.const 0.08
      local.set $v126
      local.get $v60
      local.get $v60
      f64.mul
      local.set $v133
      local.get $v59
      local.get $v59
      f64.mul
      local.set $v134
      local.get $v44
      local.get $v114
      f64.mul
      local.set $v135
      local.get $v41
      local.get $v116
      f64.mul
      local.set $v136
      local.get $v108
      local.get $v1
      f64.sub
      local.set $v177
      ;; sin via baked lookup table (see the .wasm)
      local.get $v111
      local.set $v178
      local.get $v133
      local.get $v134
      f64.add
      local.set $v186
      local.get $v135
      local.get $v136
      f64.sub
      local.set $v187
      local.get $v177
      local.get $v177
      f64.mul
      local.set $v202
      f64.const 0.72
      local.set $v210
      local.get $v178
      local.get $v210
      f64.mul
      local.set $v211
      f64.const 0.14
      local.set $v214
      local.get $v186
      local.get $v126
      f64.add
      local.set $v220
      local.get $v186
      f64.neg
      local.set $v221
      local.get $v186
      local.get $v214
      f64.add
      local.set $v222
      local.get $v109
      local.get $v202
      f64.add
      local.set $v236
      local.get $v110
      local.get $v211
      f64.add
      local.set $v237
      local.get $v220
      f64.sqrt
      local.set $v248
      f64.const 4.805000000000001
      local.set $v249
      local.get $v221
      local.get $v249
      f64.div
      local.set $v250
      f64.const 1e-05
      local.set $v271
      local.get $v236
      local.get $v271
      f64.add
      local.set $v272
      ;; cos via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v273
      ;; sin via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v275
      ;; exp via baked lookup table (see the .wasm)
      local.get $v250
      local.set $v279
      local.get $v59
      local.get $v248
      f64.div
      local.set $v280
      local.get $v279
      local.get $v222
      f64.div
      local.set $v287
      local.get $v272
      f64.sqrt
      local.set $v290
      local.get $v29
      local.get $v290
      f64.div
      local.set $v293
      local.get $v177
      local.get $v290
      f64.div
      local.set $v295
      local.get $v275
      local.get $v293
      f64.mul
      local.set $v304
      local.get $v273
      local.get $v295
      f64.mul
      local.set $v305
      local.get $v304
      local.get $v305
      f64.sub
      local.set $v309
      local.get $v309
      local.get $v82
      f64.mul
      local.set $v311
      local.get $v187
      local.get $v311
      f64.add
      local.set $v312
      local.get $v287
      local.get $v312
      f64.mul
      local.set $v313
      local.get $v280
      local.get $v313
      f64.mul
      local.set $v319
      local.get $v328
      local.get $v319
      f64.add
      local.set $v328
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_6
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_7
      (loop $sum_body_7
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_7
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v0
      local.get $v1
      f64.sub
      local.set $v29
      f64.const 2.0
      local.set $v31
      local.get $v4
      local.get $v31
      f64.mul
      local.set $v32
      f64.const 0.61
      local.set $v33
      local.get $v5
      local.get $v33
      f64.mul
      local.set $v34
      f64.const 0.83
      local.set $v35
      local.get $v6
      local.get $v35
      f64.mul
      local.set $v36
      f64.const 0.37
      local.set $v37
      local.get $v5
      local.get $v37
      f64.mul
      local.set $v38
      f64.const 0.29
      local.set $v39
      local.get $v7
      local.get $v8
      f64.add
      local.set $v40
      local.get $v9
      local.get $v10
      f64.add
      local.set $v41
      local.get $v6
      local.get $v39
      f64.mul
      local.set $v45
      local.get $v5
      local.get $v18
      f64.sub
      local.set $v70
      local.get $v6
      local.get $v16
      f64.sub
      local.set $v71
      f64.const 0.18
      local.set $v82
      local.get $v32
      local.get $v0
      f64.sub
      local.set $v108
      local.get $v29
      local.get $v29
      f64.mul
      local.set $v109
      local.get $v34
      local.get $v36
      f64.add
      local.set $v110
      local.get $v38
      local.get $v45
      f64.sub
      local.set $v111
      f64.const 1.35
      local.set $v114
      f64.const 0.3
      local.set $v116
      f64.const 0.08
      local.set $v126
      local.get $v70
      local.get $v70
      f64.mul
      local.set $v148
      local.get $v71
      local.get $v71
      f64.mul
      local.set $v149
      local.get $v41
      local.get $v114
      f64.mul
      local.set $v151
      local.get $v40
      local.get $v116
      f64.mul
      local.set $v152
      local.get $v108
      local.get $v1
      f64.sub
      local.set $v177
      ;; sin via baked lookup table (see the .wasm)
      local.get $v111
      local.set $v178
      local.get $v148
      local.get $v149
      f64.add
      local.set $v193
      local.get $v151
      local.get $v152
      f64.sub
      local.set $v194
      local.get $v177
      local.get $v177
      f64.mul
      local.set $v202
      f64.const 0.72
      local.set $v210
      local.get $v178
      local.get $v210
      f64.mul
      local.set $v211
      f64.const 0.14
      local.set $v214
      local.get $v193
      local.get $v126
      f64.add
      local.set $v226
      local.get $v193
      f64.neg
      local.set $v227
      local.get $v193
      local.get $v214
      f64.add
      local.set $v229
      local.get $v109
      local.get $v202
      f64.add
      local.set $v236
      local.get $v110
      local.get $v211
      f64.add
      local.set $v237
      local.get $v226
      f64.sqrt
      local.set $v257
      f64.const 4.805000000000001
      local.set $v258
      local.get $v227
      local.get $v258
      f64.div
      local.set $v259
      f64.const 1e-05
      local.set $v271
      local.get $v236
      local.get $v271
      f64.add
      local.set $v272
      ;; cos via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v273
      ;; sin via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v275
      local.get $v70
      local.get $v257
      f64.div
      local.set $v281
      ;; exp via baked lookup table (see the .wasm)
      local.get $v259
      local.set $v282
      local.get $v272
      f64.sqrt
      local.set $v290
      local.get $v282
      local.get $v229
      f64.div
      local.set $v291
      local.get $v29
      local.get $v290
      f64.div
      local.set $v293
      local.get $v177
      local.get $v290
      f64.div
      local.set $v295
      local.get $v273
      local.get $v293
      f64.mul
      local.set $v300
      local.get $v275
      local.get $v295
      f64.mul
      local.set $v301
      local.get $v300
      local.get $v301
      f64.add
      local.set $v308
      local.get $v308
      local.get $v82
      f64.mul
      local.set $v320
      local.get $v194
      local.get $v320
      f64.sub
      local.set $v321
      local.get $v291
      local.get $v321
      f64.mul
      local.set $v322
      local.get $v281
      local.get $v322
      f64.mul
      local.set $v323
      local.get $v329
      local.get $v323
      f64.add
      local.set $v329
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_7
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_8
      (loop $sum_body_8
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_8
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v0
      local.get $v1
      f64.sub
      local.set $v29
      f64.const 2.0
      local.set $v31
      local.get $v4
      local.get $v31
      f64.mul
      local.set $v32
      f64.const 0.61
      local.set $v33
      local.get $v5
      local.get $v33
      f64.mul
      local.set $v34
      f64.const 0.83
      local.set $v35
      local.get $v6
      local.get $v35
      f64.mul
      local.set $v36
      f64.const 0.37
      local.set $v37
      local.get $v5
      local.get $v37
      f64.mul
      local.set $v38
      f64.const 0.29
      local.set $v39
      local.get $v7
      local.get $v8
      f64.add
      local.set $v40
      local.get $v9
      local.get $v10
      f64.add
      local.set $v41
      local.get $v6
      local.get $v39
      f64.mul
      local.set $v45
      local.get $v5
      local.get $v18
      f64.sub
      local.set $v70
      local.get $v6
      local.get $v16
      f64.sub
      local.set $v71
      f64.const 0.18
      local.set $v82
      local.get $v32
      local.get $v0
      f64.sub
      local.set $v108
      local.get $v29
      local.get $v29
      f64.mul
      local.set $v109
      local.get $v34
      local.get $v36
      f64.add
      local.set $v110
      local.get $v38
      local.get $v45
      f64.sub
      local.set $v111
      f64.const 1.35
      local.set $v114
      f64.const 0.3
      local.set $v116
      f64.const 0.08
      local.set $v126
      local.get $v70
      local.get $v70
      f64.mul
      local.set $v148
      local.get $v71
      local.get $v71
      f64.mul
      local.set $v149
      local.get $v41
      local.get $v114
      f64.mul
      local.set $v151
      local.get $v40
      local.get $v116
      f64.mul
      local.set $v152
      local.get $v108
      local.get $v1
      f64.sub
      local.set $v177
      ;; sin via baked lookup table (see the .wasm)
      local.get $v111
      local.set $v178
      local.get $v148
      local.get $v149
      f64.add
      local.set $v193
      local.get $v151
      local.get $v152
      f64.sub
      local.set $v194
      local.get $v177
      local.get $v177
      f64.mul
      local.set $v202
      f64.const 0.72
      local.set $v210
      local.get $v178
      local.get $v210
      f64.mul
      local.set $v211
      f64.const 0.14
      local.set $v214
      local.get $v193
      local.get $v126
      f64.add
      local.set $v226
      local.get $v193
      f64.neg
      local.set $v227
      local.get $v193
      local.get $v214
      f64.add
      local.set $v229
      local.get $v109
      local.get $v202
      f64.add
      local.set $v236
      local.get $v110
      local.get $v211
      f64.add
      local.set $v237
      local.get $v226
      f64.sqrt
      local.set $v257
      f64.const 4.805000000000001
      local.set $v258
      local.get $v227
      local.get $v258
      f64.div
      local.set $v259
      f64.const 1e-05
      local.set $v271
      local.get $v236
      local.get $v271
      f64.add
      local.set $v272
      ;; cos via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v273
      ;; sin via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v275
      ;; exp via baked lookup table (see the .wasm)
      local.get $v259
      local.set $v282
      local.get $v71
      local.get $v257
      f64.div
      local.set $v283
      local.get $v272
      f64.sqrt
      local.set $v290
      local.get $v282
      local.get $v229
      f64.div
      local.set $v291
      local.get $v29
      local.get $v290
      f64.div
      local.set $v293
      local.get $v177
      local.get $v290
      f64.div
      local.set $v295
      local.get $v273
      local.get $v293
      f64.mul
      local.set $v300
      local.get $v275
      local.get $v295
      f64.mul
      local.set $v301
      local.get $v300
      local.get $v301
      f64.add
      local.set $v308
      local.get $v308
      local.get $v82
      f64.mul
      local.set $v320
      local.get $v194
      local.get $v320
      f64.sub
      local.set $v321
      local.get $v291
      local.get $v321
      f64.mul
      local.set $v322
      local.get $v283
      local.get $v322
      f64.mul
      local.set $v324
      local.get $v330
      local.get $v324
      f64.add
      local.set $v330
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_8
      )
    )
    i32.const 0
    local.set $i
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $entity_b_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $entity_c_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $entity_b_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $entity_c_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $entity_b_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $entity_b_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $entity_c_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $entity_c_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $v0
      local.get $v1
      f64.sub
      local.set $v29
      local.get $v2
      local.get $v3
      f64.add
      local.set $v30
      f64.const 2.0
      local.set $v31
      local.get $v4
      local.get $v31
      f64.mul
      local.set $v32
      f64.const 0.61
      local.set $v33
      local.get $v5
      local.get $v33
      f64.mul
      local.set $v34
      f64.const 0.83
      local.set $v35
      local.get $v6
      local.get $v35
      f64.mul
      local.set $v36
      f64.const 0.37
      local.set $v37
      local.get $v5
      local.get $v37
      f64.mul
      local.set $v38
      f64.const 0.29
      local.set $v39
      local.get $v7
      local.get $v8
      f64.add
      local.set $v40
      local.get $v9
      local.get $v10
      f64.add
      local.set $v41
      local.get $v6
      local.get $v11
      f64.sub
      local.set $v42
      local.get $v5
      local.get $v12
      f64.sub
      local.set $v43
      local.get $v13
      local.get $v14
      f64.add
      local.set $v44
      local.get $v6
      local.get $v39
      f64.mul
      local.set $v45
      f64.const 0.0
      local.set $v46
      local.get $v5
      local.get $v46
      f64.mul
      local.set $v47
      local.get $v6
      local.get $v46
      f64.mul
      local.set $v48
      local.get $v12
      local.get $v15
      f64.sub
      local.set $v49
      local.get $v11
      local.get $v16
      f64.sub
      local.set $v50
      local.get $v11
      local.get $v17
      f64.sub
      local.set $v51
      local.get $v12
      local.get $v18
      f64.sub
      local.set $v52
      f64.const 5.0
      local.set $v53
      local.get $v53
      local.get $v12
      f64.sub
      local.set $v54
      f64.const 0.54
      local.set $v55
      local.get $v19
      local.get $v55
      f64.mul
      local.set $v56
      f64.const 3.5
      local.set $v57
      local.get $v57
      local.get $v11
      f64.sub
      local.set $v58
      local.get $v6
      local.get $v17
      f64.sub
      local.set $v59
      local.get $v5
      local.get $v15
      f64.sub
      local.set $v60
      local.get $v20
      local.get $v55
      f64.mul
      local.set $v61
      local.get $v5
      local.get $v46
      f64.mul
      local.set $v62
      local.get $v6
      local.get $v46
      f64.mul
      local.set $v63
      local.get $v15
      local.get $v18
      f64.sub
      local.set $v64
      local.get $v17
      local.get $v16
      f64.sub
      local.set $v65
      local.get $v53
      local.get $v15
      f64.sub
      local.set $v66
      local.get $v21
      local.get $v55
      f64.mul
      local.set $v67
      local.get $v57
      local.get $v17
      f64.sub
      local.set $v68
      local.get $v22
      local.get $v55
      f64.mul
      local.set $v69
      local.get $v5
      local.get $v18
      f64.sub
      local.set $v70
      local.get $v6
      local.get $v16
      f64.sub
      local.set $v71
      local.get $v5
      local.get $v46
      f64.mul
      local.set $v72
      local.get $v6
      local.get $v46
      f64.mul
      local.set $v73
      local.get $v53
      local.get $v18
      f64.sub
      local.set $v74
      local.get $v23
      local.get $v55
      f64.mul
      local.set $v75
      local.get $v57
      local.get $v16
      f64.sub
      local.set $v76
      local.get $v24
      local.get $v55
      f64.mul
      local.set $v77
      f64.const 0.42
      local.set $v78
      local.get $v0
      local.get $v78
      f64.mul
      local.set $v79
      f64.const 0.34
      local.set $v80
      local.get $v4
      local.get $v80
      f64.mul
      local.set $v81
      f64.const 0.18
      local.set $v82
      local.get $v1
      local.get $v82
      f64.mul
      local.set $v83
      f64.const 0.35
      local.set $v84
      local.get $v25
      local.get $v84
      f64.mul
      local.set $v85
      f64.const 8.0
      local.set $v86
      local.get $v26
      local.get $v86
      f64.mul
      local.set $v87
      f64.const -0.095
      local.set $v88
      local.get $v3
      local.get $v88
      f64.mul
      local.set $v89
      f64.const 3.2
      local.set $v90
      local.get $v3
      local.get $v90
      f64.mul
      local.set $v91
      f64.const -0.072
      local.set $v92
      local.get $v3
      local.get $v92
      f64.mul
      local.set $v93
      f64.const 2.2
      local.set $v94
      local.get $v3
      local.get $v94
      f64.mul
      local.set $v95
      f64.const -0.095
      local.set $v96
      local.get $v3
      local.get $v96
      f64.mul
      local.set $v97
      local.get $v3
      local.get $v90
      f64.mul
      local.set $v98
      f64.const -0.072
      local.set $v99
      local.get $v3
      local.get $v99
      f64.mul
      local.set $v100
      local.get $v3
      local.get $v94
      f64.mul
      local.set $v101
      f64.const -0.095
      local.set $v102
      local.get $v3
      local.get $v102
      f64.mul
      local.set $v103
      local.get $v3
      local.get $v90
      f64.mul
      local.set $v104
      f64.const -0.072
      local.set $v105
      local.get $v3
      local.get $v105
      f64.mul
      local.set $v106
      local.get $v3
      local.get $v94
      f64.mul
      local.set $v107
      local.get $v32
      local.get $v0
      f64.sub
      local.set $v108
      local.get $v29
      local.get $v29
      f64.mul
      local.set $v109
      local.get $v34
      local.get $v36
      f64.add
      local.set $v110
      local.get $v38
      local.get $v45
      f64.sub
      local.set $v111
      local.get $v43
      local.get $v43
      f64.mul
      local.set $v112
      local.get $v42
      local.get $v42
      f64.mul
      local.set $v113
      f64.const 1.35
      local.set $v114
      local.get $v40
      local.get $v114
      f64.mul
      local.set $v115
      f64.const 0.3
      local.set $v116
      local.get $v44
      local.get $v116
      f64.mul
      local.set $v117
      local.get $v49
      local.get $v49
      f64.mul
      local.set $v118
      local.get $v51
      local.get $v51
      f64.mul
      local.set $v119
      local.get $v52
      local.get $v52
      f64.mul
      local.set $v120
      local.get $v50
      local.get $v50
      f64.mul
      local.set $v121
      f64.const 1.71
      local.set $v122
      local.get $v30
      local.get $v122
      f64.mul
      local.set $v123
      local.get $v49
      local.get $v116
      f64.mul
      local.set $v124
      local.get $v52
      local.get $v116
      f64.mul
      local.set $v125
      f64.const 0.08
      local.set $v126
      local.get $v54
      local.get $v126
      f64.mul
      local.set $v127
      f64.const 1.37
      local.set $v128
      local.get $v30
      local.get $v128
      f64.mul
      local.set $v129
      local.get $v51
      local.get $v116
      f64.mul
      local.set $v130
      local.get $v50
      local.get $v116
      f64.mul
      local.set $v131
      local.get $v58
      local.get $v126
      f64.mul
      local.set $v132
      local.get $v60
      local.get $v60
      f64.mul
      local.set $v133
      local.get $v59
      local.get $v59
      f64.mul
      local.set $v134
      local.get $v44
      local.get $v114
      f64.mul
      local.set $v135
      local.get $v41
      local.get $v116
      f64.mul
      local.set $v136
      local.get $v64
      local.get $v64
      f64.mul
      local.set $v137
      local.get $v65
      local.get $v65
      f64.mul
      local.set $v138
      f64.const 1.63
      local.set $v139
      local.get $v30
      local.get $v139
      f64.mul
      local.set $v140
      local.get $v49
      local.get $v116
      f64.mul
      local.set $v141
      local.get $v64
      local.get $v116
      f64.mul
      local.set $v142
      local.get $v66
      local.get $v126
      f64.mul
      local.set $v143
      f64.const 1.43
      local.set $v144
      local.get $v30
      local.get $v144
      f64.mul
      local.set $v145
      local.get $v51
      local.get $v116
      f64.mul
      local.set $v146
      local.get $v65
      local.get $v116
      f64.mul
      local.set $v147
      local.get $v70
      local.get $v70
      f64.mul
      local.set $v148
      local.get $v71
      local.get $v71
      f64.mul
      local.set $v149
      local.get $v68
      local.get $v126
      f64.mul
      local.set $v150
      local.get $v41
      local.get $v114
      f64.mul
      local.set $v151
      local.get $v40
      local.get $v116
      f64.mul
      local.set $v152
      f64.const 1.79
      local.set $v153
      local.get $v30
      local.get $v153
      f64.mul
      local.set $v154
      local.get $v52
      local.get $v116
      f64.mul
      local.set $v155
      local.get $v64
      local.get $v116
      f64.mul
      local.set $v156
      local.get $v74
      local.get $v126
      f64.mul
      local.set $v157
      f64.const 1.31
      local.set $v158
      local.get $v30
      local.get $v158
      f64.mul
      local.set $v159
      local.get $v50
      local.get $v116
      f64.mul
      local.set $v160
      local.get $v65
      local.get $v116
      f64.mul
      local.set $v161
      local.get $v76
      local.get $v126
      f64.mul
      local.set $v162
      local.get $v79
      local.get $v81
      f64.add
      local.set $v163
      f64.const 2.11
      local.set $v164
      local.get $v30
      local.get $v164
      f64.mul
      local.set $v165
      f64.const 1.91
      local.set $v166
      local.get $v30
      local.get $v166
      f64.mul
      local.set $v167
      f64.const 2.27
      local.set $v168
      local.get $v120
      local.get $v121
      f64.add
      local.set $v169
      local.get $v30
      local.get $v168
      f64.mul
      local.set $v170
      ;; exp via baked lookup table (see the .wasm)
      local.get $v89
      local.set $v171
      ;; exp via baked lookup table (see the .wasm)
      local.get $v93
      local.set $v172
      ;; exp via baked lookup table (see the .wasm)
      local.get $v97
      local.set $v173
      ;; exp via baked lookup table (see the .wasm)
      local.get $v100
      local.set $v174
      ;; exp via baked lookup table (see the .wasm)
      local.get $v103
      local.set $v175
      ;; exp via baked lookup table (see the .wasm)
      local.get $v106
      local.set $v176
      local.get $v108
      local.get $v1
      f64.sub
      local.set $v177
      ;; sin via baked lookup table (see the .wasm)
      local.get $v111
      local.set $v178
      local.get $v112
      local.get $v113
      f64.add
      local.set $v179
      local.get $v115
      local.get $v117
      f64.sub
      local.set $v180
      local.get $v118
      local.get $v119
      f64.add
      local.set $v181
      f64.const 0.2
      local.set $v182
      local.get $v123
      local.get $v182
      f64.add
      local.set $v183
      f64.const 1.1
      local.set $v184
      local.get $v129
      local.get $v184
      f64.add
      local.set $v185
      local.get $v133
      local.get $v134
      f64.add
      local.set $v186
      local.get $v135
      local.get $v136
      f64.sub
      local.set $v187
      local.get $v137
      local.get $v138
      f64.add
      local.set $v188
      f64.const 2.3
      local.set $v189
      local.get $v140
      local.get $v189
      f64.add
      local.set $v190
      f64.const 2.8
      local.set $v191
      local.get $v145
      local.get $v191
      f64.add
      local.set $v192
      local.get $v148
      local.get $v149
      f64.add
      local.set $v193
      local.get $v151
      local.get $v152
      f64.sub
      local.set $v194
      f64.const 4.2
      local.set $v195
      local.get $v154
      local.get $v195
      f64.add
      local.set $v196
      local.get $v159
      local.get $v53
      f64.add
      local.set $v197
      local.get $v163
      local.get $v83
      f64.add
      local.set $v198
      ;; sin via baked lookup table (see the .wasm)
      local.get $v165
      local.set $v199
      f64.const 2.1
      local.set $v200
      local.get $v167
      local.get $v200
      f64.add
      local.set $v201
      local.get $v177
      local.get $v177
      f64.mul
      local.set $v202
      local.get $v170
      local.get $v195
      f64.add
      local.set $v203
      local.get $v13
      local.get $v171
      f64.mul
      local.set $v204
      local.get $v14
      local.get $v172
      f64.mul
      local.set $v205
      local.get $v9
      local.get $v173
      f64.mul
      local.set $v206
      local.get $v10
      local.get $v174
      f64.mul
      local.set $v207
      local.get $v7
      local.get $v175
      f64.mul
      local.set $v208
      local.get $v8
      local.get $v176
      f64.mul
      local.set $v209
      f64.const 0.72
      local.set $v210
      local.get $v178
      local.get $v210
      f64.mul
      local.set $v211
      local.get $v179
      local.get $v126
      f64.add
      local.set $v212
      local.get $v179
      f64.neg
      local.set $v213
      f64.const 0.14
      local.set $v214
      local.get $v179
      local.get $v214
      f64.add
      local.set $v215
      local.get $v181
      local.get $v82
      f64.add
      local.set $v216
      local.get $v169
      local.get $v82
      f64.add
      local.set $v217
      ;; cos via baked lookup table (see the .wasm)
      local.get $v183
      local.set $v218
      ;; sin via baked lookup table (see the .wasm)
      local.get $v185
      local.set $v219
      local.get $v186
      local.get $v126
      f64.add
      local.set $v220
      local.get $v186
      f64.neg
      local.set $v221
      local.get $v186
      local.get $v214
      f64.add
      local.set $v222
      ;; cos via baked lookup table (see the .wasm)
      local.get $v190
      local.set $v223
      local.get $v188
      local.get $v82
      f64.add
      local.set $v224
      ;; sin via baked lookup table (see the .wasm)
      local.get $v192
      local.set $v225
      local.get $v193
      local.get $v126
      f64.add
      local.set $v226
      local.get $v193
      f64.neg
      local.set $v227
      local.get $v198
      local.get $v85
      f64.add
      local.set $v228
      local.get $v193
      local.get $v214
      f64.add
      local.set $v229
      ;; cos via baked lookup table (see the .wasm)
      local.get $v196
      local.set $v230
      ;; sin via baked lookup table (see the .wasm)
      local.get $v197
      local.set $v231
      f64.const 0.22
      local.set $v232
      local.get $v199
      local.get $v232
      f64.mul
      local.set $v233
      ;; sin via baked lookup table (see the .wasm)
      local.get $v201
      local.set $v234
      ;; sin via baked lookup table (see the .wasm)
      local.get $v203
      local.set $v235
      local.get $v109
      local.get $v202
      f64.add
      local.set $v236
      local.get $v110
      local.get $v211
      f64.add
      local.set $v237
      local.get $v212
      f64.sqrt
      local.set $v238
      f64.const 4.805000000000001
      local.set $v239
      local.get $v213
      local.get $v239
      f64.div
      local.set $v240
      f64.const 0.58
      local.set $v241
      local.get $v218
      local.get $v241
      f64.mul
      local.set $v242
      local.get $v124
      local.get $v216
      f64.div
      local.set $v243
      local.get $v125
      local.get $v217
      f64.div
      local.set $v244
      local.get $v219
      local.get $v241
      f64.mul
      local.set $v245
      local.get $v130
      local.get $v216
      f64.div
      local.set $v246
      local.get $v131
      local.get $v217
      f64.div
      local.set $v247
      local.get $v220
      f64.sqrt
      local.set $v248
      f64.const 4.805000000000001
      local.set $v249
      local.get $v221
      local.get $v249
      f64.div
      local.set $v250
      local.get $v223
      local.get $v241
      f64.mul
      local.set $v251
      local.get $v141
      local.get $v216
      f64.div
      local.set $v252
      local.get $v142
      local.get $v224
      f64.div
      local.set $v253
      local.get $v225
      local.get $v241
      f64.mul
      local.set $v254
      local.get $v146
      local.get $v216
      f64.div
      local.set $v255
      local.get $v147
      local.get $v224
      f64.div
      local.set $v256
      local.get $v226
      f64.sqrt
      local.set $v257
      f64.const 4.805000000000001
      local.set $v258
      local.get $v227
      local.get $v258
      f64.div
      local.set $v259
      local.get $v230
      local.get $v241
      f64.mul
      local.set $v260
      local.get $v155
      local.get $v217
      f64.div
      local.set $v261
      local.get $v156
      local.get $v224
      f64.div
      local.set $v262
      local.get $v231
      local.get $v241
      f64.mul
      local.set $v263
      local.get $v160
      local.get $v217
      f64.div
      local.set $v264
      local.get $v161
      local.get $v224
      f64.div
      local.set $v265
      local.get $v228
      local.get $v46
      f64.max
      local.set $v266
      f64.const 0.78
      local.set $v267
      local.get $v233
      local.get $v267
      f64.add
      local.set $v268
      local.get $v234
      local.get $v232
      f64.mul
      local.set $v269
      local.get $v235
      local.get $v232
      f64.mul
      local.set $v270
      f64.const 1e-05
      local.set $v271
      local.get $v236
      local.get $v271
      f64.add
      local.set $v272
      ;; cos via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v273
      ;; exp via baked lookup table (see the .wasm)
      local.get $v240
      local.set $v274
      ;; sin via baked lookup table (see the .wasm)
      local.get $v237
      local.set $v275
      local.get $v43
      local.get $v238
      f64.div
      local.set $v276
      local.get $v42
      local.get $v238
      f64.div
      local.set $v277
      local.get $v60
      local.get $v248
      f64.div
      local.set $v278
      ;; exp via baked lookup table (see the .wasm)
      local.get $v250
      local.set $v279
      local.get $v59
      local.get $v248
      f64.div
      local.set $v280
      local.get $v70
      local.get $v257
      f64.div
      local.set $v281
      ;; exp via baked lookup table (see the .wasm)
      local.get $v259
      local.set $v282
      local.get $v71
      local.get $v257
      f64.div
      local.set $v283
      f64.const 1.0
      local.set $v284
      local.get $v266
      local.get $v284
      f64.min
      local.set $v285
      local.get $v269
      local.get $v267
      f64.add
      local.set $v286
      local.get $v279
      local.get $v222
      f64.div
      local.set $v287
      local.get $v270
      local.get $v267
      f64.add
      local.set $v288
      local.get $v274
      local.get $v215
      f64.div
      local.set $v289
      local.get $v272
      f64.sqrt
      local.set $v290
      local.get $v282
      local.get $v229
      f64.div
      local.set $v291
      f64.const 0.12
      local.set $v292
      local.get $v29
      local.get $v290
      f64.div
      local.set $v293
      local.get $v285
      local.get $v292
      f64.mul
      local.set $v294
      local.get $v177
      local.get $v290
      f64.div
      local.set $v295
      local.get $v294
      local.get $v82
      f64.add
      local.set $v299
      local.get $v273
      local.get $v293
      f64.mul
      local.set $v300
      local.get $v275
      local.get $v295
      f64.mul
      local.set $v301
      f64.const 1e-06
      local.set $v302
      local.get $v296
      local.get $v302
      f64.add
      local.set $v303
      local.get $v275
      local.get $v293
      f64.mul
      local.set $v304
      local.get $v273
      local.get $v295
      f64.mul
      local.set $v305
      local.get $v297
      local.get $v302
      f64.add
      local.set $v306
      local.get $v298
      local.get $v302
      f64.add
      local.set $v307
      local.get $v300
      local.get $v301
      f64.add
      local.set $v308
      local.get $v304
      local.get $v305
      f64.sub
      local.set $v309
      local.get $v308
      local.get $v82
      f64.mul
      local.set $v310
      local.get $v309
      local.get $v82
      f64.mul
      local.set $v311
      local.get $v187
      local.get $v311
      f64.add
      local.set $v312
      local.get $v287
      local.get $v312
      f64.mul
      local.set $v313
      local.get $v278
      local.get $v313
      f64.mul
      local.set $v314
      local.get $v180
      local.get $v310
      f64.add
      local.set $v315
      local.get $v289
      local.get $v315
      f64.mul
      local.set $v316
      local.get $v277
      local.get $v316
      f64.mul
      local.set $v317
      local.get $v276
      local.get $v316
      f64.mul
      local.set $v318
      local.get $v280
      local.get $v313
      f64.mul
      local.set $v319
      local.get $v308
      local.get $v82
      f64.mul
      local.set $v320
      local.get $v194
      local.get $v320
      f64.sub
      local.set $v321
      local.get $v291
      local.get $v321
      f64.mul
      local.set $v322
      local.get $v281
      local.get $v322
      f64.mul
      local.set $v323
      local.get $v283
      local.get $v322
      f64.mul
      local.set $v324
      local.get $v328
      local.get $v306
      f64.div
      local.set $v331
      local.get $v329
      local.get $v307
      f64.div
      local.set $v332
      local.get $v330
      local.get $v307
      f64.div
      local.set $v333
      local.get $v326
      local.get $v303
      f64.div
      local.set $v334
      local.get $v327
      local.get $v306
      f64.div
      local.set $v335
      local.get $v325
      local.get $v303
      f64.div
      local.set $v336
      local.get $v47
      local.get $v336
      f64.add
      local.set $v337
      local.get $v48
      local.get $v334
      f64.add
      local.set $v338
      local.get $v62
      local.get $v335
      f64.add
      local.set $v339
      local.get $v63
      local.get $v331
      f64.add
      local.set $v340
      local.get $v72
      local.get $v332
      f64.add
      local.set $v341
      local.get $v73
      local.get $v333
      f64.add
      local.set $v342
      f64.const 2.35
      local.set $v343
      local.get $v337
      local.get $v343
      f64.mul
      local.set $v344
      local.get $v338
      local.get $v343
      f64.mul
      local.set $v345
      local.get $v339
      local.get $v343
      f64.mul
      local.set $v346
      local.get $v340
      local.get $v343
      f64.mul
      local.set $v347
      local.get $v341
      local.get $v343
      f64.mul
      local.set $v348
      local.get $v344
      local.get $v242
      f64.add
      local.set $v349
      local.get $v349
      local.get $v243
      f64.add
      local.set $v350
      local.get $v350
      local.get $v244
      f64.add
      local.set $v351
      local.get $v351
      local.get $v127
      f64.add
      local.set $v352
      local.get $v352
      local.get $v56
      f64.sub
      local.set $v353
      local.get $v353
      local.get $v3
      f64.mul
      local.set $v354
      local.get $v19
      local.get $v354
      f64.add
      local.set $v355
      local.get $v355
      local.get $v3
      f64.mul
      local.set $v356
      local.get $v12
      local.get $v356
      f64.add
      local.set $v357
      local.get $v345
      local.get $v245
      f64.add
      local.set $v358
      local.get $v358
      local.get $v246
      f64.add
      local.set $v359
      local.get $v359
      local.get $v247
      f64.add
      local.set $v360
      local.get $v360
      local.get $v132
      f64.add
      local.set $v361
      local.get $v361
      local.get $v61
      f64.sub
      local.set $v362
      local.get $v362
      local.get $v3
      f64.mul
      local.set $v363
      local.get $v20
      local.get $v363
      f64.add
      local.set $v364
      local.get $v364
      local.get $v3
      f64.mul
      local.set $v365
      local.get $v11
      local.get $v365
      f64.add
      local.set $v366
      local.get $v346
      local.get $v251
      f64.add
      local.set $v367
      local.get $v367
      local.get $v252
      f64.sub
      local.set $v368
      local.get $v368
      local.get $v253
      f64.add
      local.set $v369
      local.get $v369
      local.get $v143
      f64.add
      local.set $v370
      local.get $v370
      local.get $v67
      f64.sub
      local.set $v371
      local.get $v371
      local.get $v3
      f64.mul
      local.set $v372
      local.get $v21
      local.get $v372
      f64.add
      local.set $v373
      local.get $v373
      local.get $v3
      f64.mul
      local.set $v374
      local.get $v15
      local.get $v374
      f64.add
      local.set $v375
      local.get $v347
      local.get $v254
      f64.add
      local.set $v376
      local.get $v376
      local.get $v255
      f64.sub
      local.set $v377
      local.get $v377
      local.get $v256
      f64.add
      local.set $v378
      local.get $v378
      local.get $v150
      f64.add
      local.set $v379
      local.get $v379
      local.get $v69
      f64.sub
      local.set $v380
      local.get $v380
      local.get $v3
      f64.mul
      local.set $v381
      local.get $v22
      local.get $v381
      f64.add
      local.set $v382
      local.get $v382
      local.get $v3
      f64.mul
      local.set $v383
      local.get $v17
      local.get $v383
      f64.add
      local.set $v384
      local.get $v348
      local.get $v260
      f64.add
      local.set $v385
      local.get $v385
      local.get $v261
      f64.sub
      local.set $v386
      local.get $v386
      local.get $v262
      f64.sub
      local.set $v387
      local.get $v387
      local.get $v157
      f64.add
      local.set $v388
      local.get $v388
      local.get $v75
      f64.sub
      local.set $v389
      local.get $v389
      local.get $v3
      f64.mul
      local.set $v390
      local.get $v23
      local.get $v390
      f64.add
      local.set $v391
      local.get $v391
      local.get $v3
      f64.mul
      local.set $v392
      local.get $v18
      local.get $v392
      f64.add
      local.set $v393
      local.get $v342
      local.get $v343
      f64.mul
      local.set $v394
      local.get $v394
      local.get $v263
      f64.add
      local.set $v395
      local.get $v395
      local.get $v264
      f64.sub
      local.set $v396
      local.get $v396
      local.get $v265
      f64.sub
      local.set $v397
      local.get $v397
      local.get $v162
      f64.add
      local.set $v398
      local.get $v398
      local.get $v77
      f64.sub
      local.set $v399
      local.get $v399
      local.get $v3
      f64.mul
      local.set $v400
      local.get $v24
      local.get $v400
      f64.add
      local.set $v401
      local.get $v401
      local.get $v3
      f64.mul
      local.set $v402
      local.get $v16
      local.get $v402
      f64.add
      local.set $v403
      f64.const 0.65
      local.set $v404
      local.get $v357
      local.get $v404
      f64.max
      local.set $v405
      local.get $v366
      local.get $v404
      f64.max
      local.set $v406
      local.get $v375
      local.get $v404
      f64.max
      local.set $v407
      local.get $v384
      local.get $v404
      f64.max
      local.set $v408
      local.get $v393
      local.get $v404
      f64.max
      local.set $v409
      local.get $v403
      local.get $v404
      f64.max
      local.set $v410
      f64.const 9.35
      local.set $v411
      local.get $v405
      local.get $v411
      f64.min
      local.set $v412
      f64.const 6.35
      local.set $v413
      local.get $v406
      local.get $v413
      f64.min
      local.set $v414
      local.get $v407
      local.get $v411
      f64.min
      local.set $v415
      local.get $v408
      local.get $v413
      f64.min
      local.set $v416
      local.get $v409
      local.get $v411
      f64.min
      local.set $v417
      local.get $v410
      local.get $v413
      f64.min
      local.set $v418
      local.get $v6
      local.get $v418
      f64.sub
      local.set $v419
      local.get $v419
      f64.abs
      local.set $v420
      local.get $v299
      local.get $v420
      f64.sub
      local.set $v421
      local.get $v5
      local.get $v417
      f64.sub
      local.set $v422
      local.get $v422
      f64.abs
      local.set $v423
      local.get $v299
      local.get $v423
      f64.sub
      local.set $v424
      local.get $v6
      local.get $v416
      f64.sub
      local.set $v425
      local.get $v425
      f64.abs
      local.set $v426
      local.get $v299
      local.get $v426
      f64.sub
      local.set $v427
      local.get $v5
      local.get $v415
      f64.sub
      local.set $v428
      local.get $v428
      f64.abs
      local.set $v429
      local.get $v299
      local.get $v429
      f64.sub
      local.set $v430
      local.get $v6
      local.get $v414
      f64.sub
      local.set $v431
      local.get $v431
      f64.abs
      local.set $v432
      local.get $v299
      local.get $v432
      f64.sub
      local.set $v433
      local.get $v5
      local.get $v412
      f64.sub
      local.set $v434
      local.get $v434
      f64.abs
      local.set $v435
      local.get $v299
      local.get $v435
      f64.sub
      local.set $v436
      local.get $v5
      local.get $v412
      f64.sub
      local.set $v437
      local.get $v5
      local.get $v412
      f64.sub
      local.set $v438
      local.get $v437
      local.get $v438
      f64.mul
      local.set $v439
      local.get $v6
      local.get $v414
      f64.sub
      local.set $v440
      local.get $v6
      local.get $v414
      f64.sub
      local.set $v441
      local.get $v440
      local.get $v441
      f64.mul
      local.set $v442
      local.get $v439
      local.get $v442
      f64.add
      local.set $v443
      local.get $v5
      local.get $v415
      f64.sub
      local.set $v444
      local.get $v5
      local.get $v415
      f64.sub
      local.set $v445
      local.get $v444
      local.get $v445
      f64.mul
      local.set $v446
      local.get $v6
      local.get $v416
      f64.sub
      local.set $v447
      local.get $v6
      local.get $v416
      f64.sub
      local.set $v448
      local.get $v447
      local.get $v448
      f64.mul
      local.set $v449
      local.get $v446
      local.get $v449
      f64.add
      local.set $v450
      local.get $v5
      local.get $v417
      f64.sub
      local.set $v451
      local.get $v5
      local.get $v417
      f64.sub
      local.set $v452
      local.get $v451
      local.get $v452
      f64.mul
      local.set $v453
      local.get $v6
      local.get $v418
      f64.sub
      local.set $v454
      local.get $v6
      local.get $v418
      f64.sub
      local.set $v455
      local.get $v454
      local.get $v455
      f64.mul
      local.set $v456
      local.get $v453
      local.get $v456
      f64.add
      local.set $v457
      local.get $v436
      local.get $v46
      f64.max
      local.set $v458
      local.get $v433
      local.get $v46
      f64.max
      local.set $v459
      local.get $v430
      local.get $v46
      f64.max
      local.set $v460
      local.get $v427
      local.get $v46
      f64.max
      local.set $v461
      local.get $v424
      local.get $v46
      f64.max
      local.set $v462
      local.get $v457
      f64.neg
      local.set $v463
      local.get $v443
      f64.neg
      local.set $v464
      local.get $v450
      f64.neg
      local.set $v465
      local.get $v450
      f64.neg
      local.set $v466
      local.get $v443
      f64.neg
      local.set $v467
      local.get $v443
      f64.neg
      local.set $v468
      local.get $v450
      f64.neg
      local.set $v469
      local.get $v457
      f64.neg
      local.set $v470
      local.get $v458
      local.get $v459
      f64.min
      local.set $v471
      local.get $v457
      f64.neg
      local.set $v472
      local.get $v460
      local.get $v461
      f64.min
      local.set $v473
      local.get $v421
      local.get $v46
      f64.max
      local.set $v474
      local.get $v462
      local.get $v474
      f64.min
      local.set $v475
      f64.const 1.2168
      local.set $v476
      local.get $v464
      local.get $v476
      f64.div
      local.set $v477
      f64.const 1.2168
      local.set $v478
      local.get $v466
      local.get $v478
      f64.div
      local.set $v479
      f64.const 1.2168
      local.set $v480
      local.get $v463
      local.get $v480
      f64.div
      local.set $v481
      f64.const 0.1152
      local.set $v482
      local.get $v467
      local.get $v482
      f64.div
      local.set $v483
      f64.const 0.2888
      local.set $v484
      local.get $v468
      local.get $v484
      f64.div
      local.set $v485
      f64.const 0.1152
      local.set $v486
      local.get $v465
      local.get $v486
      f64.div
      local.set $v487
      f64.const 0.2888
      local.set $v488
      local.get $v469
      local.get $v488
      f64.div
      local.set $v489
      f64.const 0.1152
      local.set $v490
      local.get $v470
      local.get $v490
      f64.div
      local.set $v491
      f64.const 0.2888
      local.set $v492
      local.get $v471
      local.get $v299
      f64.div
      local.set $v493
      local.get $v473
      local.get $v299
      f64.div
      local.set $v494
      local.get $v493
      local.get $v494
      f64.max
      local.set $v495
      local.get $v475
      local.get $v299
      f64.div
      local.set $v496
      local.get $v495
      local.get $v496
      f64.max
      local.set $v497
      ;; exp via baked lookup table (see the .wasm)
      local.get $v483
      local.set $v498
      local.get $v91
      local.get $v498
      f64.mul
      local.set $v499
      local.get $v499
      local.get $v268
      f64.mul
      local.set $v500
      ;; exp via baked lookup table (see the .wasm)
      local.get $v477
      local.set $v501
      ;; exp via baked lookup table (see the .wasm)
      local.get $v479
      local.set $v502
      local.get $v501
      local.get $v502
      f64.add
      local.set $v503
      ;; exp via baked lookup table (see the .wasm)
      local.get $v481
      local.set $v504
      local.get $v503
      local.get $v504
      f64.add
      local.set $v505
      ;; exp via baked lookup table (see the .wasm)
      local.get $v485
      local.set $v506
      local.get $v95
      local.get $v506
      f64.mul
      local.set $v507
      local.get $v507
      local.get $v268
      f64.mul
      local.set $v508
      ;; exp via baked lookup table (see the .wasm)
      local.get $v487
      local.set $v509
      local.get $v98
      local.get $v509
      f64.mul
      local.set $v510
      local.get $v510
      local.get $v286
      f64.mul
      local.set $v511
      ;; exp via baked lookup table (see the .wasm)
      local.get $v489
      local.set $v512
      local.get $v101
      local.get $v512
      f64.mul
      local.set $v513
      local.get $v513
      local.get $v286
      f64.mul
      local.set $v514
      ;; exp via baked lookup table (see the .wasm)
      local.get $v491
      local.set $v515
      local.get $v104
      local.get $v515
      f64.mul
      local.set $v516
      local.get $v516
      local.get $v288
      f64.mul
      local.set $v517
      local.get $v472
      local.get $v492
      f64.div
      local.set $v518
      ;; exp via baked lookup table (see the .wasm)
      local.get $v518
      local.set $v519
      local.get $v107
      local.get $v519
      f64.mul
      local.set $v520
      local.get $v520
      local.get $v288
      f64.mul
      local.set $v521
      local.get $v505
      local.get $v284
      f64.min
      local.set $v522
      local.get $v204
      local.get $v500
      f64.add
      local.set $v523
      local.get $v497
      local.get $v232
      f64.mul
      local.set $v524
      local.get $v208
      local.get $v517
      f64.add
      local.set $v525
      local.get $v207
      local.get $v514
      f64.add
      local.set $v526
      local.get $v206
      local.get $v511
      f64.add
      local.set $v527
      local.get $v205
      local.get $v508
      f64.add
      local.set $v528
      local.get $v497
      local.get $v497
      f64.mul
      local.set $v529
      local.get $v209
      local.get $v521
      f64.add
      local.set $v530
      f64.const -0.42
      local.set $v531
      local.get $v522
      local.get $v531
      f64.mul
      local.set $v532
      local.get $v524
      local.get $v497
      f64.mul
      local.set $v533
      f64.const 54.0
      local.set $v534
      local.get $v522
      local.get $v534
      f64.mul
      local.set $v535
      f64.const 30.0
      local.set $v536
      local.get $v522
      local.get $v536
      f64.mul
      local.set $v537
      f64.const 20.0
      local.set $v538
      local.get $v522
      local.get $v538
      f64.mul
      local.set $v539
      local.get $v523
      local.get $v284
      f64.min
      local.set $v540
      local.get $v528
      local.get $v284
      f64.min
      local.set $v541
      local.get $v527
      local.get $v284
      f64.min
      local.set $v542
      local.get $v526
      local.get $v284
      f64.min
      local.set $v543
      local.get $v525
      local.get $v284
      f64.min
      local.set $v544
      local.get $v530
      local.get $v284
      f64.min
      local.set $v545
      local.get $v284
      local.get $v529
      f64.sub
      local.set $v546
      f64.const 245.0
      local.set $v547
      local.get $v529
      local.get $v547
      f64.mul
      local.set $v548
      local.get $v284
      local.get $v529
      f64.sub
      local.set $v549
      f64.const 252.0
      local.set $v550
      local.get $v529
      local.get $v550
      f64.mul
      local.set $v551
      local.get $v284
      local.get $v529
      f64.sub
      local.set $v552
      f64.const 255.0
      local.set $v553
      local.get $v529
      local.get $v553
      f64.mul
      local.set $v554
      local.get $v532
      local.get $v533
      f64.sub
      local.set $v555
      local.get $v555
      local.get $v27
      f64.sub
      local.set $v556
      local.get $v543
      local.get $v544
      f64.add
      local.set $v557
      local.get $v557
      local.get $v545
      f64.add
      local.set $v558
      local.get $v540
      local.get $v541
      f64.add
      local.set $v559
      local.get $v559
      local.get $v545
      f64.add
      local.set $v560
      local.get $v541
      local.get $v542
      f64.add
      local.set $v561
      local.get $v561
      local.get $v543
      f64.add
      local.set $v562
      local.get $v540
      local.get $v541
      f64.add
      local.set $v563
      local.get $v563
      local.get $v542
      f64.add
      local.set $v564
      local.get $v564
      local.get $v543
      f64.add
      local.set $v565
      local.get $v556
      local.get $v538
      f64.mul
      local.set $v566
      local.get $v560
      local.get $v553
      f64.mul
      local.set $v567
      local.get $v562
      local.get $v553
      f64.mul
      local.set $v568
      local.get $v565
      local.get $v544
      f64.add
      local.set $v569
      local.get $v569
      local.get $v545
      f64.add
      local.set $v570
      local.get $v566
      local.get $v87
      f64.sub
      local.set $v571
      local.get $v571
      local.get $v3
      f64.mul
      local.set $v572
      local.get $v26
      local.get $v572
      f64.add
      local.set $v573
      local.get $v558
      local.get $v553
      f64.mul
      local.set $v574
      local.get $v573
      f64.abs
      local.set $v575
      local.get $v573
      local.get $v3
      f64.mul
      local.set $v576
      local.get $v570
      local.get $v302
      f64.max
      local.set $v577
      f64.const 0.88
      local.set $v578
      local.get $v568
      local.get $v577
      f64.div
      local.set $v579
      local.get $v567
      local.get $v577
      f64.div
      local.set $v580
      local.get $v27
      local.get $v576
      f64.add
      local.set $v581
      local.get $v574
      local.get $v577
      f64.div
      local.set $v582
      local.get $v577
      local.get $v578
      f64.min
      local.set $v583
      local.get $v575
      local.get $v284
      f64.min
      local.set $v584
      local.get $v580
      local.get $v583
      f64.mul
      local.set $v585
      local.get $v284
      local.get $v583
      f64.sub
      local.set $v586
      local.get $v284
      local.get $v583
      f64.sub
      local.set $v587
      local.get $v579
      local.get $v583
      f64.mul
      local.set $v588
      local.get $v582
      local.get $v583
      f64.mul
      local.set $v589
      local.get $v581
      f64.neg
      local.set $v590
      local.get $v28
      local.get $v581
      f64.add
      local.set $v591
      local.get $v284
      local.get $v583
      f64.sub
      local.set $v592
      local.get $v584
      local.get $v86
      f64.mul
      local.set $v593
      f64.const 0.5
      local.set $v594
      local.get $v591
      local.get $v594
      f64.sub
      local.set $v595
      local.get $v590
      local.get $v78
      f64.div
      local.set $v596
      local.get $v595
      local.get $v53
      f64.div
      local.set $v597
      local.get $v596
      local.get $v46
      f64.max
      local.set $v598
      local.get $v597
      local.get $v46
      f64.max
      local.set $v599
      local.get $v598
      local.get $v284
      f64.min
      local.set $v600
      local.get $v599
      local.get $v284
      f64.min
      local.set $v601
      f64.const 34.0
      local.set $v602
      local.get $v600
      local.get $v602
      f64.mul
      local.set $v603
      f64.const 21.0
      local.set $v604
      local.get $v600
      local.get $v604
      f64.mul
      local.set $v605
      f64.const 15.0
      local.set $v606
      local.get $v600
      local.get $v606
      f64.mul
      local.set $v607
      f64.const 27.0
      local.set $v608
      local.get $v601
      local.get $v608
      f64.mul
      local.set $v609
      f64.const 18.0
      local.set $v610
      local.get $v601
      local.get $v610
      f64.mul
      local.set $v611
      f64.const 16.0
      local.set $v612
      local.get $v601
      local.get $v612
      f64.mul
      local.set $v613
      f64.const 186.0
      local.set $v614
      local.get $v609
      local.get $v614
      f64.add
      local.set $v615
      f64.const 220.0
      local.set $v616
      local.get $v611
      local.get $v616
      f64.add
      local.set $v617
      f64.const 232.0
      local.set $v618
      local.get $v613
      local.get $v618
      f64.add
      local.set $v619
      local.get $v619
      local.get $v607
      f64.add
      local.set $v620
      local.get $v620
      local.get $v539
      f64.add
      local.set $v621
      local.get $v615
      local.get $v603
      f64.sub
      local.set $v622
      local.get $v622
      local.get $v535
      f64.add
      local.set $v623
      local.get $v617
      local.get $v605
      f64.sub
      local.set $v624
      local.get $v624
      local.get $v537
      f64.add
      local.set $v625
      local.get $v623
      local.get $v46
      f64.max
      local.set $v626
      local.get $v625
      local.get $v593
      f64.add
      local.set $v627
      local.get $v621
      local.get $v46
      f64.max
      local.set $v628
      local.get $v626
      local.get $v553
      f64.min
      local.set $v629
      local.get $v627
      local.get $v46
      f64.max
      local.set $v630
      local.get $v628
      local.get $v553
      f64.min
      local.set $v631
      local.get $v629
      local.get $v586
      f64.mul
      local.set $v632
      local.get $v632
      local.get $v585
      f64.add
      local.set $v633
      local.get $v633
      local.get $v546
      f64.mul
      local.set $v634
      local.get $v634
      local.get $v548
      f64.add
      local.set $v635
      local.get $v630
      local.get $v553
      f64.min
      local.set $v636
      local.get $v636
      local.get $v587
      f64.mul
      local.set $v637
      local.get $v637
      local.get $v588
      f64.add
      local.set $v638
      local.get $v638
      local.get $v549
      f64.mul
      local.set $v639
      local.get $v639
      local.get $v551
      f64.add
      local.set $v640
      local.get $v631
      local.get $v592
      f64.mul
      local.set $v641
      local.get $v641
      local.get $v589
      f64.add
      local.set $v642
      local.get $v642
      local.get $v552
      f64.mul
      local.set $v643
      local.get $v643
      local.get $v554
      f64.add
      local.set $v644
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v635
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v640
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v644
      f64.store
      local.get $out3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v581
      f64.store
      local.get $out4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v573
      f64.store
      local.get $out5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v412
      f64.store
      local.get $out6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v414
      f64.store
      local.get $out7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v355
      f64.store
      local.get $out8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v364
      f64.store
      local.get $out9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v415
      f64.store
      local.get $out10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v416
      f64.store
      local.get $out11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v373
      f64.store
      local.get $out12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v382
      f64.store
      local.get $out13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v417
      f64.store
      local.get $out14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v418
      f64.store
      local.get $out15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v391
      f64.store
      local.get $out16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v401
      f64.store
      local.get $out17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v30
      f64.store
      local.get $out18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v540
      f64.store
      local.get $out19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v541
      f64.store
      local.get $out20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v542
      f64.store
      local.get $out21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v543
      f64.store
      local.get $out22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v544
      f64.store
      local.get $out23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v545
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)
