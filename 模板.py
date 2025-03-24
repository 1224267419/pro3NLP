# loss打印模板
#训练
for iter in range(1,n_iters+1):
    #每次训练随机选一条语句
    training_pair=tensorFromPair(random.choice(pairs))
    #获取输入和输出
    input_tensor,target_tensor=training_pair
    loss=train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,iter)

    #loss++
    print_loss_total +=loss
    plot_loss_total += loss
    #到了打印间隔
    if iter%print_every==0:

        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0#损失归0
        #打印 耗时
        print('耗时:%s (迭代步数:%d 当前迭代步数百分比:%d%%) 当前平均损失: %.4f' % (time_since(since), iter, iter / n_iters * 100, print_loss_avg))
    if iter%plot_every==0:
        #plt_losses加上损失值
        plt_losses.append(plot_loss_total / plot_every)
        plot_loss_total = 0#损失归0

#绘制损失曲线
plt.figure()
plt.plot(plt_losses)
plt.savefig("./img/s2s_loss.png")
plt.show()