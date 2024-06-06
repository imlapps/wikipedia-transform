from collections.abc import Callable, Iterable

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI
from mypy_extensions import NamedArg

from wikipedia_transform.generative_ai_pipelines import GenAiPipeline
from wikipedia_transform.models import OpenAiModel, Record, wikipedia
from wikipedia_transform.models.types import (
    EnhancementType,
    ModelQuestion,
    ModelResponse,
    RecordKey,
    RecordType,
)


class OpenAiPipeline(GenAiPipeline):
    """A concrete implementation of GenAiPipeline that uses OpenAI's generative models to enhance Records."""

    def __init__(self) -> None:
        self.__template = """
                Keep the answer as concise as possible.
                Question: {question}
                """

    def _create_question(
        self, *, record_key: RecordKey, enhancement_type: EnhancementType
    ) -> ModelQuestion:
        """Return a question for an OpenAI model."""

        match enhancement_type:
            case EnhancementType.SUMMARY:
                return f"In 5 sentences, give a summary of {record_key} based on {record_key}'s Wikipedia entry."

    def _create_chat_model(self, model: OpenAiModel) -> ChatOpenAI:
        """Return an OpenAI chat model."""

        return ChatOpenAI(name=str(model.name), temperature=model.temperature)

    def _build_chain(self, model: ChatOpenAI) -> RunnableSerializable:
        """Build a chain that consists of an OpenAI prompt, large language model and an output parser."""

        prompt = PromptTemplate.from_template(self.__template)

        return {"question": RunnablePassthrough()} | prompt | model | StrOutputParser()

    def _generate_response(
        self, *, question: ModelQuestion, chain: RunnableSerializable
    ) -> ModelResponse:
        """Invoke the OpenAI large language model and generate a response."""

        return chain.invoke(question)

    def _enhance_record(
        self,
        *,
        record: Record,
        record_type: RecordType,
        enhancement_type: EnhancementType,
        create_question: Callable[
            [
                NamedArg(RecordKey, "record_key"),
                NamedArg(EnhancementType, "enhancement_type"),
            ],
            ModelQuestion,
        ],
        create_chat_model: Callable[[OpenAiModel], ChatOpenAI],
        build_chain: Callable[[ChatOpenAI], RunnableSerializable],
        generate_response: Callable[
            [
                NamedArg(ModelQuestion, "question"),
                NamedArg(RunnableSerializable, "chain"),
            ],
            ModelResponse,
        ],
    ) -> Iterable[Record]:
        """Create a generalized workflow that yields enhanced Records."""

        if enhancement_type in record.model_fields.keys():
            match record_type:
                case RecordType.WIKIPEDIA:

                    yield wikipedia.Article(
                        **(
                            record.model_dump(by_alias=True)
                            | {
                                enhancement_type: generate_response(
                                    question=create_question(
                                        record_key=record.key,
                                        enhancement_type=enhancement_type,
                                    ),
                                    chain=build_chain(
                                        create_chat_model(OpenAiModel())),
                                )
                            }
                        )
                    )

    def enhance_record(
        self,
        *,
        record: Record,
        record_type: RecordType,
        enhancement_type: EnhancementType,
    ) -> Iterable[Record]:
        """Yield Records that have been enhanced using OpenAI models."""

        yield from self._enhance_record(
            record=record,
            record_type=record_type,
            enhancement_type=enhancement_type,
            create_question=self._create_question,
            create_chat_model=self._create_chat_model,
            build_chain=self._build_chain,
            generate_response=self._generate_response,
        )
