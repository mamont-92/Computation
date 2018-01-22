#include <QCoreApplication>
#include <chrono>
#include <QDebug>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    qDebug() << "hello SSE";

    return a.exec();
}
